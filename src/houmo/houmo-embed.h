#pragma once

#include "houmo-llmodel.h"
//
// houmo bge embedding infer
// @guoxing.xu
//
class HouMoEmbed : public HouMoLLModel::Impl {
  public:
    HouMoEmbed();
    ~HouMoEmbed();
    virtual bool houmo_load(llama_model_loader & ml) override;
    virtual void houmo_init(houmo_memory_i *memory, int seq_max = 10) override;
    virtual int
    houmo_embedding(const std::map<int32_t, std::vector<llama_token>> &batches,
                    std::map<int, float *> &embeddings) override;
    virtual int houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                              float *logits) override {
        (void)tokens; // unused parameter
        (void)seq_id; // unused parameter
        (void)logits; // unused parameter
        return 0;
    }
    virtual int houmo_decode(std::vector<llama_token> &batches,
                             std::vector<int> &seq_ids,
                             std::vector<float *> logits) override {
        (void)batches; // unused parameter
        (void)seq_ids; // unused parameter
        (void)logits;  // unused parameter
        return 0;
    }
    virtual bool is_embedding() override { return true; }

  private:
    std::shared_ptr<tcim::Module> embed_model_;
    std::map<std::string, tcim::Tensor> input_map_;
    std::map<std::string, tcim::Tensor> output_map_;
    int n_embed_ = 1024;
    int n_embedding_batch_ = 1;
};