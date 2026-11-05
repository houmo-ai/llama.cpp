#include "postmo-checkpoint-store.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace postmo {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

constexpr int32_t CHECKPOINT_FORMAT_VERSION = 2;
constexpr int32_t CHECKPOINT_FORMAT_VERSION_LEGACY = 1;

int64_t now_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string unique_suffix() {
    const auto value = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    return std::to_string(value);
}

uint64_t fnv1a_update(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hex_u64(uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

std::string file_checksum(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open checkpoint state for checksum");
    }

    uint64_t hash = 14695981039346656037ULL;
    std::vector<char> buffer(1024 * 1024);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0) {
            hash = fnv1a_update(hash, buffer.data(), static_cast<size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading checkpoint state");
    }
    return hex_u64(hash);
}

void rename_replace(const fs::path & source, const fs::path & destination) {
#if defined(_WIN32)
    if (!MoveFileExW(
            source.c_str(),
            destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::system_error(
            static_cast<int>(GetLastError()),
            std::system_category(),
            "failed to replace checkpoint file");
    }
#else
    std::error_code ec;
    fs::rename(source, destination, ec);
    if (ec) {
        throw std::runtime_error(
            "failed to rename checkpoint file: " + ec.message());
    }
#endif
}

bool valid_state_file(const std::string & value) {
    if (value.empty() || value == "." || value == "..") {
        return false;
    }
    const fs::path path(value);
    if (path.has_parent_path() || path.filename().string() != value) {
        return false;
    }
    for (unsigned char ch : value) {
        const bool valid =
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '-' || ch == '_' || ch == '.';
        if (!valid) {
            return false;
        }
    }
    return true;
}

checkpoint_record record_from_manifest(const fs::path & manifest_path) {
    checkpoint_record record;
    record.checkpoint_name = manifest_path.parent_path().filename().string();
    record.valid = false;

    try {
        std::ifstream input(manifest_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("manifest is not readable");
        }
        const json data = json::parse(input);
        record.format_version = data.at("format_version").get<int32_t>();
        record.checkpoint_name = data.at("checkpoint_name").get<std::string>();
        record.session_id = data.at("session_id").get<std::string>();
        record.model_name = data.at("model_name").get<std::string>();
        record.model_fingerprint = data.at("model_fingerprint").get<std::string>();
        record.model_desc = data.value("model_desc", std::string());
        record.context_length = data.at("context_length").get<int32_t>();
        record.id_slot = data.value("id_slot", -1);
        record.token_count = data.at("token_count").get<uint64_t>();
        record.state_size = data.at("state_size").get<uint64_t>();
        record.state_checksum = data.at("state_checksum").get<std::string>();
        record.state_file = data.value("state_file", std::string("state.bin"));
        if (!valid_state_file(record.state_file)) {
            throw std::runtime_error("invalid checkpoint state filename");
        }
        record.created_at = data.at("created_at").get<int64_t>();
        record.updated_at = data.at("updated_at").get<int64_t>();
        record.state_path = (manifest_path.parent_path() / record.state_file).string();

        std::error_code ec;
        const uint64_t actual_size = fs::file_size(record.state_path, ec);
        if (ec || actual_size != record.state_size) {
            throw std::runtime_error("checkpoint state size mismatch");
        }
        record.valid =
            record.format_version == CHECKPOINT_FORMAT_VERSION ||
            record.format_version == CHECKPOINT_FORMAT_VERSION_LEGACY;
        if (!record.valid) {
            record.error = "unsupported checkpoint format version";
        }
    } catch (const std::exception & error) {
        record.error = error.what();
    }
    return record;
}

json record_to_manifest(const checkpoint_record & record) {
    return {
        {"format_version", record.format_version},
        {"checkpoint_name", record.checkpoint_name},
        {"session_id", record.session_id},
        {"model_name", record.model_name},
        {"model_fingerprint", record.model_fingerprint},
        {"model_desc", record.model_desc},
        {"context_length", record.context_length},
        {"id_slot", record.id_slot},
        {"token_count", record.token_count},
        {"state_size", record.state_size},
        {"state_checksum", record.state_checksum},
        {"state_file", record.state_file},
        {"created_at", record.created_at},
        {"updated_at", record.updated_at},
    };
}

} // namespace

checkpoint_store::checkpoint_store(
        std::string root_path,
        checkpoint_policy policy)
    : root_path_(std::move(root_path)), policy_(std::move(policy)) {
    if (policy_.enabled() && root_path_.empty()) {
        throw std::invalid_argument(
            "named slot checkpoints require --slot-save-path");
    }
    if (policy_.enabled()) {
        std::error_code ec;
        fs::create_directories(fs::path(root_path_) / "postmo-checkpoints", ec);
        if (ec) {
            throw std::runtime_error(
                "failed to create named checkpoint root: " + ec.message());
        }
    }
}

bool checkpoint_store::enabled() const {
    return policy_.enabled();
}

const checkpoint_policy & checkpoint_store::policy() const {
    return policy_;
}

bool checkpoint_store::valid_component(const std::string & value) {
    if (value.empty() || value.size() > 128 || value == "." || value == "..") {
        return false;
    }
    for (unsigned char ch : value) {
        const bool valid =
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '-' || ch == '_' || ch == '.';
        if (!valid) {
            return false;
        }
    }
    return true;
}

std::string checkpoint_store::fingerprint(const checkpoint_scope & scope) {
    uint64_t hash = 14695981039346656037ULL;
    const std::string input = scope.model_path + '\n' + scope.model_desc + '\n' +
        std::to_string(scope.model_size) + '\n' +
        std::to_string(scope.context_length) + '\n' +
        std::to_string(scope.has_mtmd) + '\n' + scope.mmproj_path + '\n' +
        std::to_string(scope.vision_width) + 'x' +
        std::to_string(scope.vision_height);
    hash = fnv1a_update(hash, input.data(), input.size());
    return hex_u64(hash);
}

checkpoint_write checkpoint_store::prepare(
        const checkpoint_scope & scope,
        const std::string & session_id,
        const std::string & checkpoint_name) {
    if (!enabled()) {
        throw std::runtime_error("named slot checkpoints are disabled");
    }
    if (!valid_component(session_id)) {
        throw std::invalid_argument("invalid checkpoint session_id");
    }
    if (!valid_component(checkpoint_name)) {
        throw std::invalid_argument("invalid checkpoint name");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const std::string model_fingerprint = fingerprint(scope);
    const std::string key = model_fingerprint + "/" + session_id + "/" + checkpoint_name;
    if (pending_.find(key) != pending_.end()) {
        throw std::runtime_error("checkpoint update is already in progress");
    }

    const auto records = list_unlocked(scope, session_id);
    bool replacing = false;
    size_t valid_count = 0;
    for (const auto & record : records) {
        if (record.valid) {
            ++valid_count;
            replacing = replacing || record.checkpoint_name == checkpoint_name;
        }
    }
    if (!replacing && valid_count >= static_cast<size_t>(policy_.max_count)) {
        throw std::invalid_argument("named checkpoint limit reached for session");
    }

    const fs::path directory = fs::path(root_path_) / "postmo-checkpoints" /
        model_fingerprint / session_id / checkpoint_name;
    std::error_code ec;
    fs::create_directories(directory, ec);
    if (ec) {
        throw std::runtime_error(
            "failed to create checkpoint directory: " + ec.message());
    }

    checkpoint_write write;
    write.key = key;
    write.session_id = session_id;
    write.checkpoint_name = checkpoint_name;
    write.model_fingerprint = model_fingerprint;
    write.directory = directory.string();
    const std::string suffix = unique_suffix();
    write.temporary_state_path =
        (directory / ("state.tmp." + suffix)).string();
    write.final_state_path =
        (directory / ("state." + suffix + ".bin")).string();
    write.final_manifest_path = (directory / "manifest.json").string();
    pending_.insert(key);
    return write;
}

checkpoint_record checkpoint_store::commit(
        const checkpoint_write & write,
        const checkpoint_scope & scope,
        int32_t id_slot,
        uint64_t token_count,
        uint64_t state_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_.find(write.key) == pending_.end()) {
        throw std::runtime_error("checkpoint write was not prepared");
    }

    fs::path temporary_manifest;
    bool state_installed = false;
    bool manifest_committed = false;
    try {
        const fs::path temporary_state(write.temporary_state_path);
        std::error_code ec;
        const uint64_t actual_size = fs::file_size(temporary_state, ec);
        if (ec || actual_size != state_size) {
            throw std::runtime_error("saved checkpoint size mismatch");
        }

        checkpoint_record record;
        record.format_version = CHECKPOINT_FORMAT_VERSION;
        record.checkpoint_name = write.checkpoint_name;
        record.session_id = write.session_id;
        record.model_name = scope.model_name;
        record.model_fingerprint = write.model_fingerprint;
        record.model_desc = scope.model_desc;
        record.context_length = scope.context_length;
        record.id_slot = id_slot;
        record.token_count = token_count;
        record.state_size = state_size;
        record.state_checksum = file_checksum(temporary_state);
        record.state_file = fs::path(write.final_state_path).filename().string();
        record.updated_at = now_seconds();
        record.created_at = record.updated_at;

        const fs::path final_manifest(write.final_manifest_path);
        if (fs::exists(final_manifest)) {
            const auto previous = record_from_manifest(final_manifest);
            if (previous.valid) {
                record.created_at = previous.created_at;
            }
        }

        temporary_manifest =
            fs::path(write.directory) / ("manifest.json.tmp." + unique_suffix());
        {
            std::ofstream output(temporary_manifest, std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("failed to create checkpoint manifest");
            }
            output << record_to_manifest(record).dump(2) << '\n';
            output.flush();
            if (!output) {
                throw std::runtime_error("failed to write checkpoint manifest");
            }
        }

        fs::rename(temporary_state, write.final_state_path);
        state_installed = true;
        rename_replace(temporary_manifest, final_manifest);
        manifest_committed = true;
        record.state_path = write.final_state_path;
        record.valid = true;

        // The manifest is the commit point. Once it references the new state,
        // stale generations can be removed without risking the active record.
        for (fs::directory_iterator it(write.directory, ec), end; !ec && it != end; it.increment(ec)) {
            std::error_code type_ec;
            if (!it->is_regular_file(type_ec) || type_ec) {
                continue;
            }
            const std::string filename = it->path().filename().string();
            const bool is_state = filename == "state.bin" ||
                (filename.rfind("state.", 0) == 0 && it->path().extension() == ".bin");
            if (is_state && it->path() != fs::path(write.final_state_path)) {
                std::error_code remove_ec;
                fs::remove(it->path(), remove_ec);
            }
        }
        pending_.erase(write.key);
        return record;
    } catch (...) {
        pending_.erase(write.key);
        std::error_code ec;
        fs::remove(write.temporary_state_path, ec);
        if (state_installed && !manifest_committed) {
            fs::remove(write.final_state_path, ec);
        }
        if (!temporary_manifest.empty()) {
            fs::remove(temporary_manifest, ec);
        }
        throw;
    }
}

void checkpoint_store::abort(const checkpoint_write & write) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.erase(write.key);
    std::error_code ec;
    fs::remove(write.temporary_state_path, ec);
}

std::vector<checkpoint_record> checkpoint_store::list_unlocked(
        const checkpoint_scope & scope,
        const std::string & session_id) const {
    std::vector<checkpoint_record> records;
    const std::string model_fingerprint = fingerprint(scope);
    const fs::path session_path = fs::path(root_path_) / "postmo-checkpoints" /
        model_fingerprint / session_id;

    std::error_code ec;
    if (!fs::is_directory(session_path, ec)) {
        return records;
    }
    for (fs::directory_iterator it(session_path, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_directory()) {
            continue;
        }
        const fs::path manifest = it->path() / "manifest.json";
        if (fs::exists(manifest)) {
            auto record = record_from_manifest(manifest);
            if (record.valid &&
                (record.model_fingerprint != model_fingerprint ||
                 record.session_id != session_id ||
                 record.checkpoint_name != it->path().filename().string())) {
                record.valid = false;
                record.error = "checkpoint manifest identity mismatch";
            }
            records.push_back(std::move(record));
        }
    }
    return records;
}

std::vector<checkpoint_record> checkpoint_store::list(
        const checkpoint_scope & scope,
        const std::string & session_id) const {
    if (!enabled()) {
        throw std::runtime_error("named slot checkpoints are disabled");
    }
    if (!valid_component(session_id)) {
        throw std::invalid_argument("invalid checkpoint session_id");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return list_unlocked(scope, session_id);
}

checkpoint_record checkpoint_store::resolve(
        const checkpoint_scope & scope,
        const std::string & session_id,
        const std::string & checkpoint_name) const {
    if (!enabled()) {
        throw std::runtime_error("named slot checkpoints are disabled");
    }
    if (!valid_component(session_id) || !valid_component(checkpoint_name)) {
        throw std::invalid_argument("invalid checkpoint identity");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const std::string model_fingerprint = fingerprint(scope);
    const fs::path manifest = fs::path(root_path_) / "postmo-checkpoints" /
        model_fingerprint / session_id / checkpoint_name / "manifest.json";
    if (!fs::exists(manifest)) {
        throw std::runtime_error("named checkpoint not found");
    }

    auto record = record_from_manifest(manifest);
    if (!record.valid) {
        throw std::runtime_error("invalid checkpoint manifest: " + record.error);
    }
    if (record.model_fingerprint != model_fingerprint ||
        record.context_length != scope.context_length ||
        record.session_id != session_id ||
        record.checkpoint_name != checkpoint_name) {
        throw std::runtime_error("checkpoint is incompatible with the loaded model");
    }
    if (file_checksum(record.state_path) != record.state_checksum) {
        throw std::runtime_error("checkpoint checksum mismatch");
    }
    return record;
}

} // namespace postmo
