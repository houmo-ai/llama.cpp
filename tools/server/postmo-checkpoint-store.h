#pragma once

#include "postmo-checkpoint-policy.h"

#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace postmo {

struct checkpoint_scope {
    std::string model_name;
    std::string model_path;
    std::string model_desc;
    uint64_t model_size = 0;
    int32_t context_length = 0;
    bool has_mtmd = false;
    std::string mmproj_path;
    int32_t vision_width = 0;
    int32_t vision_height = 0;
};

struct checkpoint_record {
    int32_t format_version = 1;
    std::string checkpoint_name;
    std::string session_id;
    std::string model_name;
    std::string model_fingerprint;
    std::string model_desc;
    int32_t context_length = 0;
    int32_t id_slot = -1;
    uint64_t token_count = 0;
    uint64_t state_size = 0;
    std::string state_checksum;
    std::string state_file;
    int64_t created_at = 0;
    int64_t updated_at = 0;
    std::string state_path;
    bool valid = true;
    std::string error;
};

struct checkpoint_write {
    std::string key;
    std::string session_id;
    std::string checkpoint_name;
    std::string model_fingerprint;
    std::string directory;
    std::string temporary_state_path;
    std::string final_state_path;
    std::string final_manifest_path;
};

class checkpoint_store {
public:
    checkpoint_store(std::string root_path, checkpoint_policy policy);

    bool enabled() const;
    const checkpoint_policy & policy() const;

    checkpoint_write prepare(
            const checkpoint_scope & scope,
            const std::string & session_id,
            const std::string & checkpoint_name);

    checkpoint_record commit(
            const checkpoint_write & write,
            const checkpoint_scope & scope,
            int32_t id_slot,
            uint64_t token_count,
            uint64_t state_size);

    void abort(const checkpoint_write & write) noexcept;

    checkpoint_record resolve(
            const checkpoint_scope & scope,
            const std::string & session_id,
            const std::string & checkpoint_name) const;

    std::vector<checkpoint_record> list(
            const checkpoint_scope & scope,
            const std::string & session_id) const;

    static std::string fingerprint(const checkpoint_scope & scope);
    static bool valid_component(const std::string & value);

private:
    std::vector<checkpoint_record> list_unlocked(
            const checkpoint_scope & scope,
            const std::string & session_id) const;

    std::string root_path_;
    checkpoint_policy policy_;
    mutable std::mutex mutex_;
    std::set<std::string> pending_;
};

} // namespace postmo
