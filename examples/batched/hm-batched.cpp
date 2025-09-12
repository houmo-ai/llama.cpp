#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -p \"Hello my name is\" -n 32 -np 2\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;
    std::vector<std::string> prompt_list = {
        "介绍一下中国的阿里巴巴这家公司",
        "介绍一下中国的旅游城市,哪些城市适合夏天去游玩",
        "评价一下大模型带来的影响",
        "将一个100字的故事"
    };

    params.prompt = prompt_list[0];
    params.n_predict = 64;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    common_init();

    // number of parallel batches
    int n_parallel = params.n_parallel;

    // total length of the sequences including the prompt
    int n_predict = params.n_predict;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // tokenize the prompt
    if (n_parallel > static_cast<int>(prompt_list.size())) {
        LOG_ERR("n_parallel > prompt_list.size(), set n_parallel to %lu\n",
                prompt_list.size());
        n_parallel = prompt_list.size();
    }
    std::vector<std::vector<llama_token>> tokens_list_array;
    int max_tokens = 0;
    for (int i=0; i < n_parallel; i++) {
        auto prompt = prompt_list[i];
        std::vector<llama_token> tokens_list ;
        tokens_list = common_tokenize(vocab, prompt, true);
        max_tokens += tokens_list.size();
        tokens_list_array.push_back(tokens_list);
    }

    // initialize the context
    const int n_kv_req = max_tokens + (n_predict - max_tokens)*n_parallel;

    llama_context_params ctx_params = common_context_params_to_llama(params);

    std::cout << "ctx_params.n_batch=" << ctx_params.n_batch << std::endl;
    std::cout << "ctx_params.n_ubatch=" << ctx_params.n_ubatch << std::endl;
    std::cout << "ctx_params.n_seq_max=" << ctx_params.n_seq_max << std::endl;
    ctx_params.n_ctx   = n_kv_req;
    ctx_params.n_batch = std::max(n_predict, n_parallel);
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // llama_sampler_chain_add(smpl, llama_sampler_init_top_k(10));
    // llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.sampling.top_p, params.sampling.min_keep));
    // llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.2));
    // llama_sampler_chain_add(smpl, llama_sampler_init_dist (params.sampling.seed));

    if (ctx == NULL) {
        LOG_ERR("%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }
    std::vector<int32_t> i_batch;

    // create a llama_batch
    // we use this object to submit token data for decoding
    

    llama_batch batch = llama_batch_init(max_tokens, 0, n_parallel);

    std::vector<llama_seq_id> seq_ids(n_parallel, 0);

    // evaluate the initial prompt
    for (int n = 0; n < n_parallel; ++n) {
        auto &tokens_list = tokens_list_array[n];
        for (size_t i = 0; i < tokens_list.size(); ++i) {
            std::cout <<"seq_id " << n << "batch.n_tokens " << batch.n_tokens << std::endl;
            common_batch_add(batch, tokens_list[i], i, {n}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;
        i_batch.push_back(batch.n_tokens - 1);
    }

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    //// assign the system KV cache to all parallel sequences
    //// this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    //for (int32_t i = 1; i < n_parallel; ++i) {
    //    llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
    //}

    if (n_parallel > 1) {
        LOG("\n\n%s: generating %d sequences ...\n", __func__, n_parallel);
    }

    // main loop

    // we will store the parallel decoded sequences in this vector
    std::vector<std::string> streams(n_parallel);

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::cout << "====================" << batch.n_tokens - 1 <<std::endl;;
    // std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_predict) {
        // prepare the next batch
        common_batch_clear(batch);

        // sample the next token for each parallel sequence / stream
        for (int32_t i = 0; i < n_parallel; ++i) {
            std::cout<< i <<" i_batch[] " << i_batch[i] << std::endl;
            if (i_batch[i] < 0) {
                // the stream has already finished
                continue;
            }

            const llama_token new_token_id = llama_sampler_sample(smpl, ctx, i_batch[i]);

            // is it an end of generation? -> mark the stream as finished
            if (llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_predict) {
                i_batch[i] = -1;
                LOG("\n");
                if (n_parallel > 1) {
                    LOG_INF("%s: stream %d finished at n_cur = %d", __func__, i, n_cur);
                }

                continue;
            }

            // if there is only one stream, we print immediately to stdout
            if (n_parallel == 1) {
                LOG("%s", common_token_to_piece(ctx, new_token_id).c_str());
            }
            std::cout << "seq_id=" << i_batch[i]
                      << "new_token_id = " << new_token_id << " str"
                      << common_token_to_piece(ctx, new_token_id).c_str()
                      << std::endl;
            streams[i] += common_token_to_piece(ctx, new_token_id);

            i_batch[i] = batch.n_tokens;

            // push this new token for next evaluation
            common_batch_add(batch, new_token_id, n_cur, { i }, true);

            n_decode += 1;
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    //if (n_parallel > 1) {
        LOG("\n");

        for (int32_t i = 0; i < n_parallel; ++i) {
            LOG("sequence %d:\n\n%s%s\n\n", i, prompt_list[i].c_str(), streams[i].c_str());
        }
    //}

    const auto t_main_end = ggml_time_us();

    LOG_INF("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    LOG("\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    return 0;
}
