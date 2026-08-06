#include "threaded-models.h"

#include "arg.h"
#include "build-info.h"
#include "server-common.h"

#include <atomic>
#include <ctime>
#include <thread>

namespace {

void response_ok(server_http_res_ptr & res, const json & data) {
    res->status = 200;
    res->data = safe_json_to_str(data);
}

void response_error(server_http_res_ptr & res, const std::string & message, error_type type) {
    const json error = format_error_response(message, type);
    res->status = json_value(error, "code", 500);
    res->data = safe_json_to_str({{"error", error}});
}

common_params params_from_meta(const server_model_meta & meta) {
    common_params params;
    std::vector<std::string> args = meta.args;
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto & arg : args) {
        argv.push_back(arg.data());
    }
    if (!common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_SERVER)) {
        throw std::runtime_error("failed to parse preset for model " + meta.name);
    }

    // The subprocess router normally runs every preset through server.cpp,
    // which performs these normalizations before server_context::load_model().
    // The in-process router must do the same explicitly. In particular, passing
    // n_parallel == -1 into context creation can become a huge unsigned sequence
    // count and lead to an immediate OOM on the first model load.
    if (params.embedding && params.n_batch > params.n_ubatch) {
        params.n_batch = params.n_ubatch;
    }
    if (params.n_parallel < 0) {
        params.n_parallel = 1;
        params.kv_unified = true;
    }
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }
    return params;
}

server_http_context::handler_t * route_handler(server_routes & routes, const server_http_req & req, bool is_post) {
    const std::string & path = req.path;
    if (!is_post) {
        if (path == "/metrics")       return &routes.get_metrics;
        if (path == "/props")         return &routes.get_props;
        if (path == "/models" || path == "/v1/models") return &routes.get_models;
        if (path == "/lora-adapters") return &routes.get_lora_adapters;
        if (path == "/slots")         return &routes.get_slots;
        return nullptr;
    }
    if (path == "/props") return &routes.post_props;
    if (path == "/completion" || path == "/completions") return &routes.post_completions;
    if (path == "/v1/completions") return &routes.post_completions_oai;
    if (path == "/chat/completions" || path == "/v1/chat/completions") return &routes.post_chat_completions;
    if (path == "/responses" || path == "/v1/responses") return &routes.post_responses_oai;
    if (path == "/audio/transcriptions" || path == "/v1/audio/transcriptions") return &routes.post_transcriptions_oai;
    if (path == "/v1/messages") return &routes.post_anthropic_messages;
    if (path == "/v1/messages/count_tokens") return &routes.post_anthropic_count_tokens;
    if (path == "/infill") return &routes.post_infill;
    if (path == "/embedding" || path == "/embeddings") return &routes.post_embeddings;
    if (path == "/v1/embeddings") return &routes.post_embeddings_oai;
    if (path == "/rerank" || path == "/reranking" || path == "/v1/rerank" || path == "/v1/reranking") return &routes.post_rerank;
    if (path == "/tokenize") return &routes.post_tokenize;
    if (path == "/detokenize") return &routes.post_detokenize;
    if (path == "/apply-template") return &routes.post_apply_template;
    if (path == "/lora-adapters") return &routes.post_lora_adapters;
    if (path.rfind("/slots/", 0) == 0) return &routes.post_slots;
    return nullptr;
}

} // namespace

struct threaded_models::instance {
    common_params params;
    server_context context;
    server_routes routes;
    std::thread loop_thread;
    server_model_meta meta;
    std::atomic<bool> ready { false };
    std::string error;

    explicit instance(const server_model_meta & source)
        : params(params_from_meta(source)), routes(params, context), meta(source) {}

    ~instance() {
        context.terminate();
        if (loop_thread.joinable()) {
            loop_thread.join();
        }
    }
};

threaded_models::threaded_models(const common_params & params, int argc, char ** argv)
    : base_params(params), catalog(params, argc, argv, true) {
    for (const auto & meta : catalog.get_all_meta()) {
        std::string value;
        if (meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, value) &&
            common_arg_utils::is_truthy(value)) {
            auto inst = start_instance(meta);
            if (!inst->ready.load()) {
                throw std::runtime_error("failed to load startup model " + meta.name);
            }
        }
    }
}

threaded_models::~threaded_models() {
    unload_all();
}

std::string threaded_models::resolve_name(const std::string & requested) {
    auto meta = catalog.get_meta(requested);
    if (!meta) {
        throw std::invalid_argument("model '" + requested + "' not found");
    }
    return meta->name;
}

std::shared_ptr<threaded_models::instance> threaded_models::start_instance(const server_model_meta & meta) {
    auto inst = std::make_shared<instance>(meta);
    inst->meta.status = SERVER_MODEL_STATUS_LOADING;
    bool owns_load = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        auto [it, inserted] = instances.emplace(meta.name, inst);
        if (!inserted) {
            inst = it->second;
            cv.wait(lock, [&]() { return inst->meta.status != SERVER_MODEL_STATUS_LOADING; });
            return inst;
        }
        owns_load = true;
    }
    GGML_ASSERT(owns_load);

    try {
        SRV_INF("threaded model %s: n_parallel=%d, n_ctx=%d, n_batch=%d, n_ubatch=%d, kv_unified=%d\n",
                meta.name.c_str(), inst->params.n_parallel, inst->params.n_ctx,
                inst->params.n_batch, inst->params.n_ubatch, inst->params.kv_unified);
        if (!inst->context.load_model(inst->params)) {
            throw std::runtime_error("failed to load model");
        }

        inst->routes.update_meta(inst->context);
        inst->meta.loaded_info = inst->routes.get_model_info();
        inst->loop_thread = std::thread([inst]() { inst->context.start_loop(); });
        {
            std::lock_guard<std::mutex> lock(mutex);
            inst->meta.status = SERVER_MODEL_STATUS_LOADED;
            inst->meta.last_used = ggml_time_ms();
            inst->ready.store(true);
        }
        cv.notify_all();
        return inst;
    } catch (const std::exception & e) {
        std::lock_guard<std::mutex> lock(mutex);
        inst->error = e.what();
        inst->meta.status = SERVER_MODEL_STATUS_UNLOADED;
        inst->meta.exit_code = 1;
        auto it = instances.find(meta.name);
        if (it != instances.end() && it->second == inst) {
            instances.erase(it);
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex);
        inst->error = "unknown exception while loading model";
        inst->meta.status = SERVER_MODEL_STATUS_UNLOADED;
        inst->meta.exit_code = 1;
        auto it = instances.find(meta.name);
        if (it != instances.end() && it->second == inst) {
            instances.erase(it);
        }
    }
    cv.notify_all();
    return inst;
}

std::shared_ptr<threaded_models::instance> threaded_models::ensure_ready(const std::string & requested, bool autoload) {
    const std::string name = resolve_name(requested);
    {
        std::unique_lock<std::mutex> lock(mutex);
        auto it = instances.find(name);
        if (it != instances.end()) {
            auto inst = it->second;
            cv.wait(lock, [&]() { return inst->meta.status != SERVER_MODEL_STATUS_LOADING; });
            if (!inst->ready.load()) {
                const std::string error = inst->error.empty() ? "model failed to load" : inst->error;
                instances.erase(it);
                lock.unlock();
                if (!autoload) {
                    throw std::runtime_error(error);
                }
            } else {
                return inst;
            }
        }
    }
    if (!autoload) {
        throw std::invalid_argument("model is not loaded");
    }

    auto all = catalog.get_meta(name);
    if (!all) {
        throw std::invalid_argument("model not found");
    }

    std::string lru_name;
    if (base_params.models_max > 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if ((int) instances.size() >= base_params.models_max) {
            int64_t oldest = INT64_MAX;
            for (const auto & entry : instances) {
                if (entry.second->meta.status == SERVER_MODEL_STATUS_LOADED &&
                    entry.second->meta.last_used < oldest) {
                    oldest = entry.second->meta.last_used;
                    lru_name = entry.first;
                }
            }
        }
    }
    if (!lru_name.empty()) {
        stop_instance(lru_name);
    } else if (base_params.models_max > 0) {
        std::lock_guard<std::mutex> lock(mutex);
        if ((int) instances.size() >= base_params.models_max) {
            throw std::runtime_error("model limit reached while all instances are loading");
        }
    }
    auto inst = start_instance(*all);
    if (!inst->ready.load()) {
        throw std::runtime_error(inst->error.empty() ? "model failed to load" : inst->error);
    }
    return inst;
}

server_http_res_ptr threaded_models::dispatch(const server_http_req & req, bool is_post) {
    std::string requested;
    if (is_post) {
        requested = json_value(json::parse(req.body), "model", std::string());
    } else {
        requested = req.get_param("model");
    }
    if (requested.empty()) {
        auto res = std::make_unique<server_http_res>();
        response_error(res, "model name is missing from the request", ERROR_TYPE_INVALID_REQUEST);
        return res;
    }
    const std::string autoload_arg = req.get_param("autoload");
    const bool autoload = autoload_arg.empty() ? base_params.models_autoload : common_arg_utils::is_truthy(autoload_arg);
    auto inst = ensure_ready(requested, autoload);
    auto * handler = route_handler(inst->routes, req, is_post);
    if (handler == nullptr || !*handler) {
        auto res = std::make_unique<server_http_res>();
        response_error(res, "route is not supported by threaded router", ERROR_TYPE_NOT_SUPPORTED);
        return res;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        inst->meta.last_used = ggml_time_ms();
    }
    auto res = (*handler)(req);
    res->lifetime_guard = inst;
    return res;
}

server_http_res_ptr threaded_models::get_props(const server_http_req & req) {
    if (!req.get_param("model").empty()) {
        return dispatch(req, false);
    }
    auto res = std::make_unique<server_http_res>();
    json ui_settings = json::object();
    const std::string & ui_config = !base_params.ui_config_json.empty()
        ? base_params.ui_config_json
        : base_params.webui_config_json;
    if (!ui_config.empty()) {
        ui_settings = json::parse(ui_config);
    }
    response_ok(res, {
        // The Web UI intentionally performs an exact role comparison. Keep the
        // standard router role while exposing the implementation separately.
        {"role", "router"},
        {"router_implementation", "threaded"},
        {"max_instances", base_params.models_max},
        {"models_autoload", base_params.models_autoload},
        {"model_alias", "llama-threaded-server"},
        {"model_path", "none"},
        {"default_generation_settings", {
            {"params", json::object()},
            {"n_ctx", 0},
        }},
        {"ui_settings", ui_settings},
        {"webui_settings", ui_settings},
        {"build_info", std::string(llama_build_info())},
        {"cors_proxy_enabled", base_params.ui_mcp_proxy || base_params.webui_mcp_proxy},
    });
    return res;
}

server_http_res_ptr threaded_models::get_models(const server_http_req & req) {
    if (!req.get_param("reload").empty()) {
        catalog.load_models();
    }
    json data = json::array();
    const std::time_t created = std::time(nullptr);
    for (auto meta : catalog.get_all_meta()) {
        std::shared_ptr<instance> inst;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = instances.find(meta.name);
            if (it != instances.end()) inst = it->second;
        }
        if (inst) {
            std::lock_guard<std::mutex> lock(mutex);
            meta = inst->meta;
        }
        json item = {
            {"id", meta.name}, {"aliases", meta.aliases}, {"tags", meta.tags},
            {"object", "model"}, {"owned_by", "llamacpp"}, {"created", created},
            {"status", {{"value", server_model_status_to_string(meta.status)}, {"args", meta.args}}},
        };
        if (inst && inst->ready.load()) {
            for (auto it = meta.loaded_info.begin(); it != meta.loaded_info.end(); ++it) {
                if (!item.contains(it.key())) item[it.key()] = it.value();
            }
        }
        data.push_back(std::move(item));
    }
    auto res = std::make_unique<server_http_res>();
    response_ok(res, {{"data", data}, {"object", "list"}});
    return res;
}

server_http_res_ptr threaded_models::load_model(const server_http_req & req) {
    const std::string requested = json_value(json::parse(req.body), "model", std::string());
    ensure_ready(requested, true);
    auto res = std::make_unique<server_http_res>();
    response_ok(res, {{"success", true}});
    return res;
}

void threaded_models::stop_instance(const std::string & requested) {
    const std::string name = resolve_name(requested);
    std::shared_ptr<instance> inst;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = instances.find(name);
        if (it == instances.end()) return;
        inst = it->second;
        instances.erase(it);
    }
    inst->ready.store(false);
    inst->context.terminate();
    if (inst->loop_thread.joinable()) inst->loop_thread.join();
    inst->meta.status = SERVER_MODEL_STATUS_UNLOADED;
}

server_http_res_ptr threaded_models::unload_model(const server_http_req & req) {
    const std::string requested = json_value(json::parse(req.body), "model", std::string());
    stop_instance(requested);
    auto res = std::make_unique<server_http_res>();
    response_ok(res, {{"success", true}});
    return res;
}

void threaded_models::unload_all() {
    std::vector<std::shared_ptr<instance>> active;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & entry : instances) active.push_back(entry.second);
        instances.clear();
    }
    for (auto & inst : active) inst->context.terminate();
    for (auto & inst : active) {
        if (inst->loop_thread.joinable()) inst->loop_thread.join();
    }
}
