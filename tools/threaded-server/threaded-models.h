#pragma once

#include "server-context.h"
#include "server-models.h"

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>

namespace postmo {
class device_arbiter;
}

struct threaded_models {
    explicit threaded_models(const common_params & params, int argc, char ** argv);
    ~threaded_models();

    server_http_res_ptr dispatch(const server_http_req & req, bool is_post);
    server_http_res_ptr get_props(const server_http_req & req);
    server_http_res_ptr get_models(const server_http_req & req);
    server_http_res_ptr load_model(const server_http_req & req);
    server_http_res_ptr unload_model(const server_http_req & req);

    void unload_all();

private:
    struct instance;

    common_params base_params;
    server_models catalog;
    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, std::shared_ptr<instance>> instances;
    std::shared_ptr<postmo::device_arbiter> device_arbiter;

    std::string resolve_name(const std::string & requested);
    std::shared_ptr<instance> ensure_ready(const std::string & name, bool autoload);
    std::shared_ptr<instance> start_instance(const server_model_meta & meta);
    void stop_instance(const std::string & name);
};
