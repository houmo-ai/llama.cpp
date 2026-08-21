#pragma once

#include <cstdint>
#include <set>
#include <string>

namespace postmo {

enum class checkpoint_trigger {
    explicit_request,
    interval,
    prefill_end,
    request_end,
};

struct checkpoint_policy {
    int32_t max_count = 0;
    int32_t every_n_tokens = 0;
    std::set<checkpoint_trigger> triggers;

    bool enabled() const;
    bool has_trigger(checkpoint_trigger trigger) const;

    static checkpoint_policy parse(
            int32_t max_count,
            int32_t every_n_tokens,
            const std::string & trigger_list);
};

} // namespace postmo
