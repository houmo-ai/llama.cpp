#include "postmo-checkpoint-policy.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace postmo {
namespace {

std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

checkpoint_trigger parse_trigger(const std::string & value) {
    if (value == "explicit") {
        return checkpoint_trigger::explicit_request;
    }
    if (value == "interval") {
        return checkpoint_trigger::interval;
    }
    if (value == "prefill_end") {
        return checkpoint_trigger::prefill_end;
    }
    if (value == "request_end") {
        return checkpoint_trigger::request_end;
    }
    throw std::invalid_argument("unknown slot checkpoint trigger: " + value);
}

} // namespace

bool checkpoint_policy::enabled() const {
    return max_count > 0;
}

bool checkpoint_policy::has_trigger(checkpoint_trigger trigger) const {
    return triggers.find(trigger) != triggers.end();
}

checkpoint_policy checkpoint_policy::parse(
        int32_t max_count,
        int32_t every_n_tokens,
        const std::string & trigger_list) {
    if (max_count < 0) {
        throw std::invalid_argument("slot checkpoint max count must be >= 0");
    }
    if (every_n_tokens < 0) {
        throw std::invalid_argument("slot checkpoint interval must be >= 0");
    }

    checkpoint_policy result;
    result.max_count = max_count;
    result.every_n_tokens = every_n_tokens;

    std::istringstream input(trigger_list);
    std::string item;
    while (std::getline(input, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            result.triggers.insert(parse_trigger(item));
        }
    }
    if (result.triggers.empty()) {
        result.triggers.insert(checkpoint_trigger::explicit_request);
    }
    if (result.every_n_tokens > 0) {
        result.triggers.insert(checkpoint_trigger::interval);
    }
    if (result.has_trigger(checkpoint_trigger::interval) ||
        result.has_trigger(checkpoint_trigger::prefill_end) ||
        result.has_trigger(checkpoint_trigger::request_end)) {
        throw std::invalid_argument(
            "automatic slot checkpoint triggers are not implemented yet; "
            "use --slot-checkpoint-every-n-tokens 0 and --slot-checkpoint-triggers explicit");
    }
    return result;
}

} // namespace postmo
