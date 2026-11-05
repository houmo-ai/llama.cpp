#include "postmo-checkpoint-policy.h"
#include "postmo-checkpoint-store.h"
#include "postmo-device-arbiter.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct temporary_directory {
    fs::path path = fs::temp_directory_path() /
        ("postmo-server-test-" + std::to_string(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()));

    temporary_directory() {
        fs::create_directories(path);
    }

    ~temporary_directory() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void write_state(const postmo::checkpoint_write & write, const std::string & value) {
    std::ofstream output(write.temporary_state_path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "failed to create temporary state");
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    require(static_cast<bool>(output), "failed to write temporary state");
}

void wait_for_waiters(postmo::device_arbiter & arbiter, size_t count) {
    for (int retry = 0; retry < 1000; ++retry) {
        if (arbiter.stats().waiting == count) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("timed out waiting for device arbiter waiters");
}

void test_device_arbiter_priority() {
    postmo::device_arbiter arbiter;
    auto blocker = arbiter.acquire(100);
    std::mutex order_mutex;
    std::vector<int32_t> order;

    std::thread low([&]() {
        auto permit = arbiter.acquire(20);
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(20);
    });
    wait_for_waiters(arbiter, 1);
    std::thread high([&]() {
        auto permit = arbiter.acquire(1);
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(1);
    });
    wait_for_waiters(arbiter, 2);

    blocker = {};
    low.join();
    high.join();

    require(order == std::vector<int32_t> {1, 20}, "priority ordering failed");
    require(arbiter.stats().acquisitions == 3, "arbiter acquisition count mismatch");
}

void test_device_arbiter_fifo() {
    postmo::device_arbiter arbiter;
    auto blocker = arbiter.acquire(0);
    std::mutex order_mutex;
    std::vector<int32_t> order;

    std::thread first([&]() {
        auto permit = arbiter.acquire(10);
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(1);
    });
    wait_for_waiters(arbiter, 1);
    std::thread second([&]() {
        auto permit = arbiter.acquire(10);
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(2);
    });
    wait_for_waiters(arbiter, 2);

    blocker = {};
    first.join();
    second.join();
    require(order == std::vector<int32_t> {1, 2}, "same-priority FIFO ordering failed");
}

void test_checkpoint_store() {
    temporary_directory root;
    const auto policy = postmo::checkpoint_policy::parse(2, 0, "explicit");
    postmo::checkpoint_store store(root.path.string(), policy);
    const postmo::checkpoint_scope scope {
        "primary-alias", "models/model.gguf", "test model", 1024, 4096,
    };
    auto alias_scope = scope;
    alias_scope.model_name = "renamed-alias";
    auto vision_scope = scope;
    vision_scope.has_mtmd = true;
    vision_scope.mmproj_path = "models/mmproj.gguf";
    vision_scope.vision_width = 448;
    vision_scope.vision_height = 448;

    auto system = store.prepare(scope, "session-1", "system_prompt_end");
    write_state(system, "system-state");
    const auto system_record = store.commit(system, scope, 0, 10, 12);
    require(system_record.valid, "system checkpoint is invalid");
    require(system_record.format_version == 2, "checkpoint format version mismatch");

    auto latest = store.prepare(alias_scope, "session-1", "session_latest");
    require(latest.model_fingerprint == system.model_fingerprint, "model alias changed fingerprint");
    auto vision = store.prepare(vision_scope, "session-2", "system_prompt_end");
    require(vision.model_fingerprint != system.model_fingerprint, "VIT runtime signature did not change fingerprint");
    store.abort(vision);
    write_state(latest, "latest-v1");
    const auto latest_v1 = store.commit(latest, alias_scope, 1, 20, 9);

    bool limit_rejected = false;
    try {
        (void) store.prepare(scope, "session-1", "third");
    } catch (const std::invalid_argument &) {
        limit_rejected = true;
    }
    require(limit_rejected, "third named checkpoint was not rejected");

    auto replacement = store.prepare(scope, "session-1", "session_latest");
    write_state(replacement, "latest-v2-longer");
    const auto latest_v2 = store.commit(replacement, scope, 1, 30, 16);
    require(latest_v1.state_path != latest_v2.state_path, "checkpoint state generation was reused");
    require(!fs::exists(latest_v1.state_path), "stale checkpoint generation was not removed");
    require(fs::exists(latest_v2.state_path), "active checkpoint generation is missing");

    const auto records = store.list(scope, "session-1");
    require(records.size() == 2, "checkpoint list size mismatch");
    const auto resolved = store.resolve(scope, "session-1", "session_latest");
    require(resolved.token_count == 30, "resolved checkpoint token count mismatch");

    {
        std::fstream output(resolved.state_path, std::ios::binary | std::ios::in | std::ios::out);
        require(static_cast<bool>(output), "failed to open checkpoint for corruption test");
        output.seekp(0);
        output.put('X');
    }
    bool corruption_rejected = false;
    try {
        (void) store.resolve(scope, "session-1", "session_latest");
    } catch (const std::runtime_error &) {
        corruption_rejected = true;
    }
    require(corruption_rejected, "checkpoint checksum corruption was not rejected");
}

void test_automatic_checkpoint_policy_is_rejected() {
    bool rejected = false;
    try {
        (void) postmo::checkpoint_policy::parse(2, 128, "explicit,interval");
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "unsupported automatic checkpoint policy was accepted");
}

} // namespace

int main() {
    test_device_arbiter_priority();
    test_device_arbiter_fifo();
    test_checkpoint_store();
    test_automatic_checkpoint_policy_is_rejected();
    return 0;
}
