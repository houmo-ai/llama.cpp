#include "threaded-models.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "server-context.h"
#include "server-http.h"

#include <atomic>
#include <clocale>
#include <csignal>

static std::function<void(int)> shutdown_handler;
static std::atomic_flag terminating = ATOMIC_FLAG_INIT;

static void signal_handler(int signal) {
    if (terminating.test_and_set()) {
        std::_Exit(1);
    }
    shutdown_handler(signal);
}

static server_http_context::handler_t safe_handler(server_http_context::handler_t handler) {
    return [handler = std::move(handler)](const server_http_req & req) -> server_http_res_ptr {
        try {
            return handler(req);
        } catch (const std::invalid_argument & e) {
            auto res = std::make_unique<server_http_res>();
            const json error = format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST);
            res->status = json_value(error, "code", 400);
            res->data = safe_json_to_str({{"error", error}});
            return res;
        } catch (const std::exception & e) {
            auto res = std::make_unique<server_http_res>();
            const json error = format_error_response(e.what(), ERROR_TYPE_SERVER);
            res->status = json_value(error, "code", 500);
            res->data = safe_json_to_str({{"error", error}});
            return res;
        }
    };
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    if (!params.model.path.empty()) {
        LOG_ERR("llama-threaded-server is a router-only custom executable; use --models-dir or --models-preset\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    common_params_print_info(params, false);

    server_http_context http;
    if (!http.init(params)) {
        LOG_ERR("failed to initialize HTTP server\n");
        llama_backend_free();
        return 1;
    }

    // Reuse the standard router health handler. This context never loads a model.
    server_context empty_context;
    server_routes base_routes(params, empty_context);
    threaded_models models(params, argc, argv);

    auto proxy_get  = [&models](const server_http_req & req) { return models.dispatch(req, false); };
    auto proxy_post = [&models](const server_http_req & req) { return models.dispatch(req, true); };

    http.get("/health", safe_handler(base_routes.get_health));
    http.get("/v1/health", safe_handler(base_routes.get_health));
    http.get("/metrics", safe_handler(proxy_get));
    http.get("/props", safe_handler([&models](const server_http_req & req) { return models.get_props(req); }));
    http.get("/models", safe_handler([&models](const server_http_req & req) { return models.get_models(req); }));
    http.get("/v1/models", safe_handler([&models](const server_http_req & req) { return models.get_models(req); }));
    http.get("/lora-adapters", safe_handler(proxy_get));
    http.get("/slots", safe_handler(proxy_get));

    http.post("/models/load", safe_handler([&models](const server_http_req & req) { return models.load_model(req); }));
    http.post("/models/unload", safe_handler([&models](const server_http_req & req) { return models.unload_model(req); }));
    http.post("/props", safe_handler(proxy_post));
    http.post("/completion", safe_handler(proxy_post));
    http.post("/completions", safe_handler(proxy_post));
    http.post("/v1/completions", safe_handler(proxy_post));
    http.post("/chat/completions", safe_handler(proxy_post));
    http.post("/v1/chat/completions", safe_handler(proxy_post));
    http.post("/responses", safe_handler(proxy_post));
    http.post("/v1/responses", safe_handler(proxy_post));
    http.post("/audio/transcriptions", safe_handler(proxy_post));
    http.post("/v1/audio/transcriptions", safe_handler(proxy_post));
    http.post("/v1/messages", safe_handler(proxy_post));
    http.post("/v1/messages/count_tokens", safe_handler(proxy_post));
    http.post("/infill", safe_handler(proxy_post));
    http.post("/embedding", safe_handler(proxy_post));
    http.post("/embeddings", safe_handler(proxy_post));
    http.post("/v1/embeddings", safe_handler(proxy_post));
    http.post("/rerank", safe_handler(proxy_post));
    http.post("/reranking", safe_handler(proxy_post));
    http.post("/v1/rerank", safe_handler(proxy_post));
    http.post("/v1/reranking", safe_handler(proxy_post));
    http.post("/tokenize", safe_handler(proxy_post));
    http.post("/detokenize", safe_handler(proxy_post));
    http.post("/apply-template", safe_handler(proxy_post));
    http.post("/lora-adapters", safe_handler(proxy_post));
    http.post("/slots/:id_slot", safe_handler(proxy_post));

    http.register_gcp_compat();
    if (!http.start()) {
        llama_backend_free();
        return 1;
    }
    http.is_ready.store(true);
    shutdown_handler = [&](int) { http.stop(); };

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction action {};
    action.sa_handler = signal_handler;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
#endif

    LOG_INF("threaded router is listening on %s\n", http.listening_address.c_str());
    if (http.thread.joinable()) {
        http.thread.join();
    }
    models.unload_all();
    llama_backend_free();
    return 0;
}

