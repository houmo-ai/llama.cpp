#include "houmo-llmodel.h"
#include "ggml-hmutils.h"
#include "gguf.h"
#include "houmo-embed.h"
#include "houmo-gptoss.h"
#include "houmo-qwenllm.h"
#include "houmo-qwen35llm.h"
#include "houmo-qwen35mtp.h"
#include "houmo-vllmodel.h"
#include "houmo-glmocrmodel.h"
#include "houmo-gemma4.h"
#include "houmo-gemma4mtp.h"
#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using json = nlohmann::ordered_json;

namespace {

int hmm_group_id(const std::string &name) {
    static const std::regex suffix("_([0-9]+)\\.hmm$");
    std::smatch match;
    return std::regex_search(name, match, suffix) ? std::stoi(match[1]) : 0;
}

int embedding_npu_device_id() {
    const char *env = std::getenv("HOUMO_DEVICE_ID");
    return env ? std::stoi(env) : 0;
}

std::vector<int> collect_hmm_groups(const llama_model_loader &loader, bool mtp) {
    std::set<int> prefill_groups;
    std::set<int> decode_groups;
    gguf_context *ctx = loader.metadata_ptr.get();
    const std::regex prefill(mtp ? "^prefill_mtp(_\\d+)?\\.hmm$"
                                 : "^prefill(_\\d+)?\\.hmm$");
    const std::regex decode(mtp ? "^decode(r)?_mtp(_\\d+)?\\.hmm$"
                                : "^decode(r)?(_\\d+)?\\.hmm$");
    const std::regex assistant("^assistant\\.hmm(s)?$");
    for (int i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const char *raw_name = gguf_get_tensor_name(ctx, i);
        if (!raw_name) {
            continue;
        }
        const std::string name(raw_name);
        if (std::regex_match(name, prefill)) {
            prefill_groups.insert(hmm_group_id(name));
        } else if (std::regex_match(name, decode) ||
                   (mtp && std::regex_match(name, assistant))) {
            decode_groups.insert(hmm_group_id(name));
        }
    }
    if (mtp && prefill_groups.empty()) {
        prefill_groups = decode_groups;
    }
    if (!decode_groups.empty() && prefill_groups != decode_groups) {
        LLAMA_LOG_ERROR("prefill/decode HMM group sets do not match\n");
        return {};
    }
    for (int group_id : prefill_groups) {
        if (!decode_groups.empty() && decode_groups.count(group_id) == 0) {
            LLAMA_LOG_ERROR("missing decode HMM for group %d\n", group_id);
            return {};
        }
    }
    return {prefill_groups.begin(), prefill_groups.end()};
}

class GroupMemory final : public houmo_memory_i {
  public:
    GroupMemory(houmo_memory_i *memory, int group_index, int group_count)
        : memory_(memory), group_index_(group_index), group_count_(group_count) {}

    bool seq_rm(llama_seq_id id, llama_pos p0, llama_pos p1) override {
        return memory_->seq_rm(global(id), p0, p1);
    }
    void seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) override {
        memory_->seq_cp(global(src), global(dst), p0, p1);
    }
    void seq_keep(llama_seq_id id) override { memory_->seq_keep(global(id)); }
    void seq_add(llama_seq_id id, llama_pos p0, llama_pos p1, llama_pos shift) override {
        memory_->seq_add(global(id), p0, p1, shift);
    }
    void seq_div(llama_seq_id id, llama_pos p0, llama_pos p1, int d) override {
        memory_->seq_div(global(id), p0, p1, d);
    }
    llama_pos seq_pos_min(llama_seq_id id) const override {
        return memory_->seq_pos_min(global(id));
    }
    llama_pos seq_pos_max(llama_seq_id id) const override {
        return memory_->seq_pos_max(global(id));
    }

    void state_write(llama_io_write_i &io, llama_seq_id id = -1,
                     llama_state_seq_flags flags = 0) const override {
        memory_->state_write(io, global(id), flags);
    }
    void state_read(llama_io_read_i &io, llama_seq_id id = -1,
                    llama_state_seq_flags flags = 0) override {
        memory_->state_read(io, global(id), flags);
    }

  private:
    llama_seq_id global(llama_seq_id local) const {
        return local * group_count_ + group_index_;
    }
    houmo_memory_i *memory_;
    int group_index_;
    int group_count_;
};

HouMoLLModel::Impl *create_impl(const std::string &arch, bool mtp,
                               HouMoLLModel::Impl *target = nullptr) {
    if (arch.find("gemma4") != std::string::npos) {
        return mtp ? static_cast<HouMoLLModel::Impl *>(
                         new HouMoGemma4MTP(static_cast<HouMoGemma4 *>(target)))
                   : static_cast<HouMoLLModel::Impl *>(new HouMoGemma4());
    }
    if (mtp) {
        return new HoumoQwen35MTP();
    }
    if (arch.find("glm4") != std::string::npos) return new HouMoGLMOCRModel();
    if (arch.find("vl") != std::string::npos) return new HouMoVLLModel();
    if (arch.find("gpt-oss") != std::string::npos) return new HouMoGPTOSS();
    if (arch.find("bert") != std::string::npos) return new HouMoEmbed();
    if (arch.find("qwen35") != std::string::npos ||
        arch.find("qwen3next") != std::string::npos) return new HoumoQwen35LLM();
    if (arch.find("qwen") != std::string::npos) return new HoumoQwenLLM();
    return nullptr;
}

} // namespace

namespace {
constexpr uint32_t HOUMO_STATE_MAGIC = 0x484d5354; // HMST
constexpr uint32_t HOUMO_STATE_VERSION = 1;

struct houmo_state_header {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;
    int32_t source_seq_id;
    int32_t group_id;
    int32_t local_seq_id;
};
} // namespace

void houmo_memory_i::state_write_impl(llama_io_write_i &io,
                                      llama_seq_id seq_id,
                                      llama_state_seq_flags flags) const {
    model_->state_write(io, seq_id, flags, is_mtp_);
}

void houmo_memory_i::state_read_impl(llama_io_read_i &io,
                                     llama_seq_id seq_id,
                                     llama_state_seq_flags flags) {
    model_->state_read(io, seq_id, flags, is_mtp_);
}

void HouMoLLModel::Impl::state_write(llama_io_write_i &, int,
                                     llama_state_seq_flags) const {
    throw std::runtime_error("Houmo state serialization is not supported by this model");
}

void HouMoLLModel::Impl::state_read(llama_io_read_i &, int,
                                    llama_state_seq_flags) {
    throw std::runtime_error("Houmo state restore is not supported by this model");
}

static bool string_replace_once(std::string &str, const std::string &from,
                                const std::string &to) {
    if (from.empty())
        return false;
    size_t pos = str.find(from);
    if (pos == std::string::npos)
        return false;
    str.replace(pos, from.length(), to);
    return true;
}

static bool gguf_has_mtp_tensors(const llama_model_loader &loader) {
    if (!loader.metadata_ptr) {
        return false;
    }
    gguf_context *ctx = loader.metadata_ptr.get();
    const int n_tensors = gguf_get_n_tensors(ctx);
    const std::regex kPrefillMtpPattern("^prefill_mtp(_\\d+)?\\.hmm$");
    const std::regex kDecodeMtpPattern("^decode(r)?_mtp(_\\d+)?\\.hmm$");
    const std::regex kAssistantPattern("^assistant\\.hmm(s)?$");
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        if (name == nullptr) {
            continue;
        }
        if (std::regex_match(name, kPrefillMtpPattern) ||
            std::regex_match(name, kDecodeMtpPattern) ||
            std::regex_match(name, kAssistantPattern)) {
            return true;
        }
    }
    return false;
}

static bool is_kv_cache_io_name(const std::string &name) {
    return name.find("past_key_cache") != std::string::npos ||
           name.find("past_value_cache") != std::string::npos ||
           name.find("shared_key_cache") != std::string::npos ||
           name.find("shared_value_cache") != std::string::npos ||
           name.find("_self_attn_vcache") != std::string::npos ||
           name.find("_self_attn_kcache") != std::string::npos ||
           name.find("conv_cache_out") != std::string::npos ||
           name.find("recurrent_state_out") != std::string::npos ||
           name.find("past_conv_cache") != std::string::npos ||
           name.find("past_recurrent_state") != std::string::npos;
}

std::shared_ptr<tcim::Module> HouMoLLModel::Impl::create_infer_engine(
    bool prefill, void *raw_data, int64_t size, bool load_from_file,
    std::vector<std::string> dummy_tensor_names) {
    if (!wm_configured_) {
        const std::string backend = "Xh2HalBackend";
        tcim::DevManager dev_manager = tcim::DevManager::Create(device_ids_, backend);
        wm_ = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
        wm_configured_ = true;
    }
    tcim::Status status;
    auto infer_model = std::make_shared<tcim::Module>();
    auto option = tcim::Module::Option(wm_);
    option.SetDummyTensors(dummy_tensor_names);
    option.EnableHostLazyLoading(enable_lazy_mode_);
    if (prefill) {
        if (!load_from_file)
            status = infer_model->LoadModel(raw_data, size, option);
        else {
            std::string prefill_name((char *)raw_data, size);
            string_replace_once(prefill_name, "-00001-", "-00002-");
            LLAMA_LOG_INFO("LoadModel prefill_name: %s\n",
                           prefill_name.c_str());
            status = infer_model->LoadModel(prefill_name, option);
        }
    } else {
        if (!load_from_file)
            status = infer_model->LoadModel(raw_data, size, option);
        else {
            std::string decoder_name((char *)raw_data, size);
            string_replace_once(decoder_name, "-00001-", "-00003-");
            LLAMA_LOG_INFO("LoadModel decoder_name: %s\n",
                           decoder_name.c_str());
            status = infer_model->LoadModel(decoder_name, option);
        }
    }
    if (status != tcim::Status::OK) {
        LLAMA_LOG_ERROR("LoadModel failed: prefill == %d\n", prefill);
        return nullptr;
    }
    return infer_model;
}

std::shared_ptr<tcim::Module> HouMoLLModel::Impl::create_infer_engine(
    const std::string &file_name, size_t offset, size_t size,
    std::vector<std::string> dummy_tensor_names) {
    if (!wm_configured_) {
        const std::string backend = "Xh2HalBackend";
        set_weight_manager_handle(houmo_weight_manager_acquire(
                file_name, device_ids_, backend, {}));
    }
    tcim::Status status;
    auto infer_model = std::make_shared<tcim::Module>();
    auto option = tcim::Module::Option(wm_);
    option.SetDummyTensors(dummy_tensor_names);
    option.EnableHostLazyLoading(enable_lazy_mode_);
    option.SetModelOffset(offset, size);
    status = infer_model->LoadModel(file_name, option);
    if (status != tcim::Status::OK) {
        LLAMA_LOG_ERROR("LoadModel failed: status == %d\n", (int)status);
        return nullptr;
    }
    return infer_model;
}

bool HouMoLLModel::Impl::houmo_load(llama_model_loader &ml, std::vector<int> device_ids) {
    std::string file_name = ml.fname_;
    arch_name = ml.arch_name;
    device_ids_ = device_ids;
    gguf_context *ctx = ml.metadata_ptr.get();
    parser_gguf_parameters(ctx, ml.arch_name);
    const bool load_mtp_modules = wants_mtp_modules();
    // 从 gguf 文件中加载 hmmodels
    const std::regex kPrefillPattern("^prefill(_\\d+)?\\.hmm$");
    const std::regex kPrefillMtpPattern("^prefill_mtp(_\\d+)?\\.hmm$");
    const std::regex kDecodePattern("^decode(r)?(_\\d+)?\\.hmm$");
    const std::regex kAssistantPattern("^assistant\\.hmm(s)?$");
    const std::regex kDecodeMtpPattern("^decode(r)?_mtp(_\\d+)?\\.hmm$");
    // 环境变量中是否开启了LLAMA_ENABLE_LAZY_MODE
    enable_lazy_mode_ = !ml.use_mmap;
    if (const char *env = std::getenv("LLAMA_ENABLE_LAZY_MODE")) {
        enable_lazy_mode_ = (std::string(env) == "1");
    }
    LLAMA_LOG_INFO("LLAMA_ENABLE_LAZY_MODE: %d\n", enable_lazy_mode_);
    // --embed-npu: 环境变量或 CLI 参数控制
    {
        const char *env = std::getenv("LLAMA_ARG_EMBED_NPU");
        embed_npu_ = (env && std::string(env) == "1");
    }
    const int embedding_npu_device =
        embed_npu_ ? embedding_npu_device_id() : -1;
    std::vector<std::string> dummy_tensor_names;
    // 防止更高效使用内存，这个将prefill 设置为dummy_tensor_names
    for (int idx = 0; idx < n_blocks_; idx++) {
        std::string k_cache_name =
            "model_layers_" + std::to_string(idx) + "_self_attn_kcache_input";
        std::string v_cache_name =
            "model_layers_" + std::to_string(idx) + "_self_attn_vcache_input";
        dummy_tensor_names.push_back(k_cache_name);
        dummy_tensor_names.push_back(v_cache_name);
    }
    parse_deviceids();
    std::vector<houmo_content_range> embedding_ranges;
    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char *raw_name = gguf_get_tensor_name(ctx, i);
        if (raw_name == nullptr) {
            continue;
        }
        const std::string name(raw_name);
        const int64_t tensor_size = static_cast<int64_t>(gguf_get_tensor_size(ctx, i));
        const int64_t tensor_offset = static_cast<int64_t>(
                gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx));
        houmo_content_range range{name, tensor_offset, tensor_size};
        if (name.compare(0, strlen("quant_embedding.bin"),
                         "quant_embedding.bin") == 0) {
            embedding_ranges.push_back(range);
        }
    }
    const std::string embedding_identity = houmo_content_identity(file_name, embedding_ranges);
    if (!wm_configured_) {
        set_weight_manager_handle(houmo_weight_manager_acquire(
                file_name, device_ids_, "Xh2HalBackend",
                embedding_identity));
    }
    // 1. 加载 embedding
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        LLAMA_LOG_INFO("tensor[%d]: name = %s\n", i, name);
        const size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset =
            gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        if (strncmp(name, "quant_embedding.bin",
                    strlen("quant_embedding.bin")) == 0) {
            if (embedding_layer_ == nullptr)
                embedding_layer_ = houmo_embedding_layer_acquire(
                        file_name, offset, size, embedding_identity,
                        embedding_npu_device);
        } else if (std::regex_match(name, kPrefillMtpPattern)) {
            if (hmm_group_id(name) != load_group_id_) continue;
            if (!load_mtp_modules) {
                LLAMA_LOG_INFO("%s: skip prefill_mtp.hmm for target impl\n", __func__);
                continue;
            }
            std::vector<std::string> mtp_dummy_tensor_names = dummy_tensor_names;
            // MTP 草稿 prefill：与普通 prefill 共用 prefill_model_ 槽位
            // （同一个 HouMoLLModel 实例只承载一个 gguf，要么是 target 要么是 MTP）。
            prefill_model_ = create_infer_engine(file_name, offset, size,
                                                 mtp_dummy_tensor_names);
            if (prefill_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load prefill_mtp.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill_mtp.hmm (MTP draft module)\n",
                           __func__);
        } else if (std::regex_match(name, kPrefillPattern)) {
            if (hmm_group_id(name) != load_group_id_) continue;
            if (load_mtp_modules) {
                LLAMA_LOG_INFO("%s: skip prefill.hmm for MTP impl\n", __func__);
                continue;
            }
            // load prefill
            const std::vector<std::string> prefill_dummy_tensor_names =
                arch_name.find("gemma4") != std::string::npos
                    ? std::vector<std::string>{}
                    : dummy_tensor_names;
            prefill_model_ = create_infer_engine(file_name, offset, size,
                                                 prefill_dummy_tensor_names);
            if (prefill_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load prefill.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load prefill.hmm\n", __func__);
        } else if (std::regex_match(name, kDecodeMtpPattern) ||
                   std::regex_match(name, kAssistantPattern)) {
            if (hmm_group_id(name) != load_group_id_) continue;
            if (!load_mtp_modules) {
                LLAMA_LOG_INFO("%s: skip decode_mtp.hmm for target impl\n", __func__);
                continue;
            }
            std::vector<std::string> mtp_dummy_tensor_names;
            if (arch_name.find("gemma4") != std::string::npos) {
                mtp_dummy_tensor_names = {
                    "shared_key_cache_sliding",
                    "shared_value_cache_sliding",
                    "shared_key_cache_full",
                    "shared_value_cache_full",
                };
            }
            decode_model_ = create_infer_engine(file_name, offset, size,
                                                mtp_dummy_tensor_names);
            if (decode_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load decode_mtp.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load decode_mtp.hmm (MTP draft module)\n",
                           __func__);
        } else if (std::regex_match(name, kDecodePattern)) {
            if (hmm_group_id(name) != load_group_id_) continue;
            if (load_mtp_modules) {
                LLAMA_LOG_INFO("%s: skip decode.hmm for MTP impl\n", __func__);
                continue;
            }
            // load decode
            const std::vector<std::string> decode_dummy_tensor_names =
                arch_name.find("gemma4") != std::string::npos
                    ? dummy_tensor_names
                    : std::vector<std::string>{};
            decode_model_ = create_infer_engine(file_name, offset, size,
                                                decode_dummy_tensor_names);

            if (decode_model_ == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to load decode.hmm\n", __func__);
                return false;
            }
            LLAMA_LOG_INFO("%s: succeed to load decode.hmm\n", __func__);
        } else {
            LLAMA_LOG_INFO("tensor[%d]: name = %s, size = %zu, skipped\n", i,
                           name, size);
        }
    }
    if (n_split_count_ == 3) {
        prefill_model_ =
            create_infer_engine(true, const_cast<char *>(file_name.c_str()),
                                file_name.length(), true);
        decode_model_ =
            create_infer_engine(false, const_cast<char *>(file_name.c_str()),
                                file_name.length(), true);
    }
    if (load_mtp_modules && arch_name.find("gemma4") != std::string::npos) {
        if (decode_model_ == nullptr || embedding_layer_ == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to load gemma4 MTP model. load "
                            "decode_mtp=%d, load embedding=%d\n",
                            __func__, (decode_model_ == nullptr),
                            (embedding_layer_ == nullptr));
            return false;
        }
        return true;
    }
    if (prefill_model_ == nullptr || embedding_layer_ == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to load model. load "
                        "prefill=%d,load decode=%d\n",
                        __func__, (prefill_model_ == nullptr),
                        (embedding_layer_ == nullptr));
        return false;
    }

    return true;
}

bool HouMoLLModel::Impl::supports_seq_max(int seq_max, std::string &reason) const {
    if (seq_max <= 0) {
        reason = "seq_max must be positive";
        return false;
    }
    if (seq_max > kSeqMax) {
        reason = "seq_max exceeds backend kSeqMax=" + std::to_string(kSeqMax);
        return false;
    }
    if (decode_model_ != nullptr && n_decode_batch_ > 1 &&
        seq_max > n_decode_batch_) {
        reason = "logical local sequence count exceeds physical decode batch slots=" +
                 std::to_string(n_decode_batch_);
        return false;
    }
    return true;
}

void HouMoLLModel::Impl::model_inout_init(int seq_max) {
    n_vocab_ = embedding_layer_->get_vocab_size();
    // init prefill
    prefill_input_init();
    // 填充prefill 的output
    prefill_output_init();
    // init decode
    decode_input_init();
    // 填充decode 的output
    decode_output_init();
    // init kv cache
    GGML_ASSERT(seq_max > 0 && seq_max <= kSeqMax);
    GGML_ASSERT(decode_model_ == nullptr || n_decode_batch_ == 1 ||
                seq_max <= n_decode_batch_);
    int size = (seq_max + n_prefill_batch_ - 1) / n_prefill_batch_;
    size = std::max(size, n_decode_batch_);
    if (is_mtp_model()) {
        // MTP draft HMM does not expose/use the target model KV cache fields.
        // Its state is driven by hidden/input_embedding/position length inputs,
        // so the generic per-layer KV initializer must not probe target KV names.
        kv_cache_.clear();
        need_set_kv_cache_ = 0;
        LLAMA_LOG_INFO("%s: skip generic KV cache init for MTP model\n", __func__);
    } else {
        initialize_kv_cache(size);
    }
    LLAMA_LOG_INFO("%s: HMM graph init success\n", __func__);
    LLAMA_LOG_INFO("prefill_length_ = %d\n context_length_ = %d\n "
                   "n_blocks_ = %d\n n_embd_ = %ld\n n_prefill_batch_ = %d\n "
                   "n_decode_batch_ = "
                   "%d\n kv_cache_size = %zu\n need_set_kv_cache_ = "
                   "%d\n seq_max = %d\n",
                   prefill_length_, context_length_, n_blocks_, n_embd_,
                   n_prefill_batch_, n_decode_batch_, kv_cache_.size(),
                   need_set_kv_cache_, seq_max);
}

void HouMoLLModel::Impl::prefill_input_init() {
    if (prefill_model_ == nullptr) {
        return;
    }
    if (prefill_input_map_.size() > 0) {
        return;
    }
    for (size_t idx = 0; idx < prefill_model_->GetInputNum(); idx++) {
        auto input_name = prefill_model_->GetInputName(idx);
        LLAMA_LOG_DEBUG("%s: prefill input_name: %s\n", __func__,
                        input_name.c_str());
        if (input_name.find("past_key_cache_") != std::string::npos) {
            kCacheKeyWord = "past_key_cache_";
        }
        if (is_kv_cache_io_name(input_name)) {
            continue;
        }
        input_names_.push_back(input_name);
        auto input_info =
            prefill_model_->GetInputInfo(input_name).AsContiguous();
        auto shape = input_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: prefill input_shape: %ld, type = %d\n",
                           input_name.c_str(), shape[i],
                           (int)input_info.DataType());
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        prefill_input_map_.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
    // 填充prefill_length_ & n_prefill_batch_ only support batch_size = 1
    n_prefill_batch_ = 1;
    prefill_length_ =
        prefill_input_map_[prefill_model_->GetInputName(0)].Info().Shape()[1] *
        prefill_input_map_[prefill_model_->GetInputName(0)].Info().Shape()[0];
}

void HouMoLLModel::Impl::prefill_output_init() {
    if (prefill_model_ == nullptr) {
        return;
    }
    if (prefill_output_map_.size() > 0) {
        return;
    }
    int output_num = prefill_model_->GetOutputNum();
    for (int idx = 0; idx < output_num; idx++) {
        auto output_name = prefill_model_->GetOutputName(idx);
        LLAMA_LOG_INFO("%s: prefill output_name: %s\n", __func__,
                       output_name.c_str());
        if (is_kv_cache_io_name(output_name)) {
            continue;
        }
        auto output_info = prefill_model_->GetOutputInfo(output_name)
                               .AsContiguous()
                               .AsType(tcim::DataType::FLOAT32);
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        prefill_output_map_.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        prefill_output_names_.push_back(output_name);
    }
}

void HouMoLLModel::Impl::decode_input_init() {
    if (decode_model_ == nullptr) {
        return;
    }
    if (decode_input_map_.size() > 0) {
        return;
    }

    size_t input_num_decode = decode_model_->GetInputNum();
    for (size_t idx = 0; idx < input_num_decode; idx++) {
        auto input_name = decode_model_->GetInputName(idx);
        LLAMA_LOG_DEBUG("%s: decode_model input_name: %s\n", __func__,
                        input_name.c_str());
        if (input_name.find("past_key_cache_") != std::string::npos) {
            kCacheKeyWord = "past_key_cache_";
        }
        if (is_kv_cache_io_name(input_name)) {
            continue;
        }
        if (input_name.find("valid_length") != std::string::npos ||
            input_name.find("past_seq_length") != std::string::npos) {
            decode_valid_length_names_.push_back(input_name);
        }
        if (input_name.find("current_length") != std::string::npos ||
            input_name.find("current_input_length") != std::string::npos) {
            decode_current_length_names_.push_back(input_name);
        }
        LLAMA_LOG_INFO("%s: decode input_name: %s\n", __func__,
                       input_name.c_str());
        auto input_info =
            decode_model_->GetInputInfo(input_name).AsContiguous();
        auto shape = input_info.Shape();
        for (size_t i = 0; i < shape.size(); i++) {
            LLAMA_LOG_INFO("%s: decode input_shape: %ld\n", __func__, shape[i]);
        }
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        decode_input_map_.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
    n_decode_batch_ = decode_input_map_[decode_model_->GetInputName(0)].Info().Shape()[0];
}

void HouMoLLModel::Impl::decode_output_init() {
    // 填充decode 的output
    if (decode_model_ == nullptr) {
        return;
    }
    if (decode_output_map_.size() > 0) {
        return;
    }
    int output_num_decode = decode_model_->GetOutputNum();
    for (int idx = 0; idx < output_num_decode; idx++) {
        auto output_name = decode_model_->GetOutputName(idx);
        LLAMA_LOG_INFO("%s: decode output_name: %s\n", __func__,
                       output_name.c_str());
        if (is_kv_cache_io_name(output_name)) {
            continue;
        }
        auto output_info = decode_model_->GetOutputInfo(output_name)
                               .AsContiguous()
                               .AsType(tcim::DataType::FLOAT32);
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        decode_output_map_.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        decode_output_names_.push_back(output_name);
    }
}

void HouMoLLModel::Impl::parser_gguf_parameters(struct gguf_context *ctx,
                                                std::string &model_arch) {
    std::string key = model_arch + kBlockCountEx;
    auto key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                       key.c_str());
    } else {
        n_blocks_ = gguf_get_val_u32(ctx, key_id);
    }
    key = model_arch + kEmbeddingLengthEx;
    key_id = gguf_find_key(ctx, key.c_str());
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                       key.c_str());
    } else {
        n_embd_ = gguf_get_val_u32(ctx, key_id);
    }
    key = "tokenizer.ggml.tokens";
    key_id = gguf_find_key(ctx, key.c_str());
    if (key_id >= 0) {
        n_vocab_ = (int64_t) gguf_get_arr_n(ctx, key_id);
    }
    auto key_n_split = gguf_find_key(ctx, kSpliteCount.c_str());
    if (key_n_split < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                       kSpliteCount.c_str());
    } else {
        n_split_count_ = gguf_get_val_i32(ctx, key_n_split);
    }

    key_id = gguf_find_key(ctx, "general.name");
    if (key_id >= 0) {
        std::string general_name = gguf_get_val_str(ctx, key_id);
        std::transform(general_name.begin(), general_name.end(),
                       general_name.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        asr_force_aligner_ =
            general_name.find("forcealigner") != std::string::npos ||
            general_name.find("forcedaligner") != std::string::npos ||
            general_name.find("forced aligner") != std::string::npos;
    }

    // 解析hmm.info
    key_id = gguf_find_key(ctx, kHmmInfo.c_str());
    if (key_id < 0) {
        LLAMA_LOG_INFO("%s: not found key %s in gguf file\n", __func__,
                       kHmmInfo.c_str());
        return;
    }
    auto hmm_info_str = gguf_get_val_str(ctx, key_id);
    LLAMA_LOG_INFO("%s: hmm_info: %s\n", __func__, hmm_info_str);
    try {
        // 解析hmm.info
        auto hmm_info = json::parse(hmm_info_str);
        if (hmm_info.contains("device_num")) {
            device_num_ = hmm_info["device_num"];
        }

    } catch (const json::parse_error &e) {
        LLAMA_LOG_ERROR("%s: failed to parse hmm_info: %s\n", __func__,
                        e.what());
        return;
    }
}

void HouMoLLModel::Impl::read_tensor_to_buffer(const std::string &file_name,
                                               size_t offset, size_t size,
                                               std::vector<char> &buffer) {
    if (endsWithMem(file_name)) {
        void **buffers = nullptr;
        void *bufferSize = nullptr;
        int buffers_count = 0;
        if (parseMemoryString(file_name.c_str(), &buffers, &bufferSize,
                              &buffers_count)) {
            LLAMA_LOG_INFO("buffers (void**): 0x%p\n", (void *)buffers);
            LLAMA_LOG_INFO("bufferSize (void*): 0x%p\n", bufferSize);
            LLAMA_LOG_INFO("buffers_count (int): %d\n", buffers_count);
        }
        size_t model_size = *(size_t *)bufferSize;
        LLAMA_LOG_INFO("buffers[0]: %p\n", buffers[0]);
        LLAMA_LOG_INFO("model_size: %zu\n", model_size);
        memcpy(buffer.data(), (char *)buffers[0] + offset, size);
        LLAMA_LOG_INFO("read membuffer offset:%zu  size:%zu to buffer\r\n",
                       offset, size);
        return;
    }

    LLAMA_LOG_INFO("fread mem offset:%zu  size:%zu to buffer\r\n", offset,
                   size);

    std::ifstream input_file(file_name, std::ios::binary);
    if (!input_file.is_open()) {
        LLAMA_LOG_ERROR("Failed to open file: %s\n", file_name.c_str());
        return;
    }

    // 移动到偏移位置
    input_file.seekg(offset, std::ios::beg);
    if (input_file.fail()) {
        LLAMA_LOG_ERROR("Failed to seek to offset: %zu in file: %s\n", offset,
                        file_name.c_str());
        input_file.close();
        return;
    }

    // 读取数据
    input_file.read(buffer.data(), size);
    if (input_file.fail()) {
        LLAMA_LOG_ERROR("Failed to read %zu bytes from file: %s\n", size,
                        file_name.c_str());
        input_file.close();
        return;
    }
}

void HouMoLLModel::Impl::parse_deviceids() {
    const char *device_id_str = std::getenv("HOUMO_DEVICE_ID");
    if (device_id_str == nullptr) {
        if (device_ids_.size() == 0)
            device_ids_.push_back(0);
    } else {
        LLAMA_LOG_INFO("HOUMO_DEVICE_ID: %s\n", device_id_str);
        device_ids_.clear();
        std::string deviceList(device_id_str);
        std::istringstream iss(deviceList);
        std::string token;

        while (std::getline(iss, token, ',')) {
            try {
                // 转换为整数并添加到列表
                device_ids_.push_back(std::stoi(token));
            } catch (const std::invalid_argument &) {
                LLAMA_LOG_ERROR("Invalid device ID: %s\n", token.c_str());
            } catch (const std::out_of_range &) {
                LLAMA_LOG_ERROR("Device ID out of range: %s\n", token.c_str());
            }
        }
    }
    // 校验设置的是否合法
    if (device_num_ == 2 && device_ids_.size() == 1) {
        // 单卡模型，但是模型是双卡模型，需要将device_ids 扩展为{0,1}
        device_ids_ = {0, 1};
        LLAMA_LOG_WARN("dual card model, need to set device_ids to {0,1}\n");
    } else if (device_num_ == 1 && device_ids_.size() > 1) {
        // 模型是单卡模型，需要将device_ids 缩小为{0}
        device_ids_ = {0};
        LLAMA_LOG_WARN("single card model, set device_ids size > 1, need to "
                       "resize device_ids to {0}\n");
    } else if (device_num_ == 4 && device_ids_.size() == 1) {
        // 模型是四卡模型，需要将device_ids 扩展为{0,1,2,3}
        device_ids_ = {0, 1, 2, 3};
        LLAMA_LOG_WARN("4 card model, need to set device_ids to {0,1,2,3}");
    } else {
        LLAMA_LOG_INFO("run models in %zu cards", device_ids_.size());
    }
}

void HouMoLLModel::Impl::set_input_data(tcim::Tensor &tensor, int batch_size,
                                        int value) {
    if (tensor.Info().DataType() == tcim::DataType::INT16) {
        std::vector<int16_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT32) {
        std::vector<int32_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT8) {
        std::vector<int8_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::FLOAT16) {
        std::vector<uint16_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else {
        throw std::invalid_argument("Unsupported tensor data type");
    }
}

void HouMoLLModel::Impl::set_input_data(tcim::Tensor &tensor,
                                        std::vector<int32_t> &int32_vec,
                                        size_t offset) {
    if (offset >= int32_vec.size()) {
        LLAMA_LOG_ERROR(
            "%s: offset is out of range, offset: %zu, int32_vec.size(): %zu\n",
            __func__, offset, int32_vec.size());
        throw std::out_of_range("Offset is out of range");
    }
    if (tensor.Info().DataType() == tcim::DataType::INT16) {
        std::vector<int16_t> data_vec(int32_vec.size() - offset, 0);
        for (size_t i = offset; i < int32_vec.size(); i++) {
            data_vec[i - offset] = static_cast<int16_t>(int32_vec[i]);
        }
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT32) {
        std::vector<int32_t> data_vec(int32_vec.size() - offset, 0);
        for (size_t i = offset; i < int32_vec.size(); i++) {
            data_vec[i - offset] = static_cast<int32_t>(int32_vec[i]);
        }
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else {
        throw std::invalid_argument("Unsupported tensor data type");
    }
}
void HouMoLLModel::Impl::initialize_kv_cache(int size) {
    if (decode_model_ == nullptr) {
        // 非decode模型, 无需初始化kv_cache
        std::string name =
            prefill_model_->GetInputName(prefill_input_map_.size());
        context_length_ =
            std::max(prefill_model_->GetInputInfo(name).Shape()[2],
                     prefill_model_->GetInputInfo(name).Shape()[3]);
        return;
    }
    std::string kv_prefix_name = kCacheKeyWord;
    // 检查是否有.hmcc.format 后缀. 开发编译的模型会添加后缀
    std::string ends_with = "";
    if (prefill_model_->GetInputName(0).find(".hmcc.format") !=
                     std::string::npos) {
        ends_with = ".hmcc.format";
    }
    int kv_cache_count = n_blocks_;
    if (arch_name.find("qwen35") != std::string::npos || arch_name.find("qwen3next") != std::string::npos) {
        kv_cache_count = n_blocks_ / 4;
    }
    if ((size_t)size != kv_cache_.size()) {
        LLAMA_LOG_INFO("%s: initialize kv cache, size = %d\n", __func__, size);
        kv_cache_.clear();
    }
    for (int i = 0; i < size; i++) {
        kv_cache_item item;
        item.seq_id = i;
        for (int idx = 0; idx < kv_cache_count; idx++) {
            std::string k_cache_name = kv_prefix_name + std::to_string(idx) +
                                       "_self_attn_kcache_input" + ends_with;
            std::string v_cache_name = kv_prefix_name + std::to_string(idx) +
                                       "_self_attn_vcache_input" + ends_with;
            if (kCacheKeyWord == "past_key_cache_") {
                k_cache_name = "past_key_cache_" + std::to_string(idx);
                v_cache_name = "past_value_cache_" + std::to_string(idx);
            }
            item.names.push_back(k_cache_name);
            item.names.push_back(v_cache_name);
            LLAMA_LOG_DEBUG("%s: kv_cache_name: %s, n_blocks = %d\n", __func__, k_cache_name.c_str(), n_blocks_);
            if (n_decode_batch_ > 1) {
                if (i > 0) {
                    k_cache_name = k_cache_name + "_batch" + std::to_string(i);
                    v_cache_name = v_cache_name + "_batch" + std::to_string(i);
                }
            }
            const bool prefill_owned_gemma4_cache =
                arch_name.find("gemma4") != std::string::npos &&
                prefill_model_ != nullptr;
            auto kcache = prefill_owned_gemma4_cache
                              ? prefill_model_->GetDevInput(k_cache_name)
                              : decode_model_->GetDevInput(k_cache_name);
            auto vcache = prefill_owned_gemma4_cache
                              ? prefill_model_->GetDevInput(v_cache_name)
                              : decode_model_->GetDevInput(v_cache_name);
            if (prefill_owned_gemma4_cache) {
                zero_tensor(kcache);
                zero_tensor(vcache);
            }
            if (n_decode_batch_ > 1 || size == 1) {
                item.tensors.push_back(kcache);
                item.tensors.push_back(vcache);
            } else {
                item.tensors.push_back(kcache.Clone());
                item.tensors.push_back(vcache.Clone());
                need_set_kv_cache_ = 1;
            }
            context_length_ = std::max(
                kcache.Info().Shape()[kcache.Info().Shape().size() - 1],
                kcache.Info().Shape()[kcache.Info().Shape().size() - 2]);
        }
        kv_cache_[i] = item;
    }
    if (arch_name.find("gemma4") != std::string::npos) {
        // Gemma4 decode KV inputs were loaded as dummy tensors and must always
        // be rebound to the prefill-owned storage before verify/decode runs.
        need_set_kv_cache_ = 1;
    }
}

void HouMoLLModel::Impl::set_kv_cache_for_prefill(int seq_id) {
    if (kv_cache_.empty()) {
        return;
    }
    const auto item = kv_cache_.find(seq_id);
    if (item == kv_cache_.end()) {
        LLAMA_LOG_ERROR("kv_cache_.find(%d) failed.\n", seq_id);
        return;
    }
    for (size_t idx = 0; idx < item->second.tensors.size(); idx++) {
        std::string name = item->second.names[idx];
        auto tensor = item->second.tensors[idx];
        prefill_model_->SetInput(name, tensor);
    }
}

void HouMoLLModel::Impl::set_kv_cache_for_decode(
    const std::vector<int> &seq_ids) {
    if (need_set_kv_cache_ != 0) {
        for (auto seq_id : seq_ids) {
            auto &item = kv_cache_[seq_id];
            for (size_t idx = 0; idx < item.tensors.size(); idx++) {
                std::string name = item.names[idx];
                if (n_decode_batch_ > 1)
                    name = name + "_batch" + std::to_string(item.seq_id);
                auto tensor = item.tensors[idx];
                decode_model_->SetInput(name, tensor);
            }
        }
    }
}

int HouMoLLModel::Impl::set_input_for_decode(std::vector<llama_token> &batches,
                                             std::vector<int> &seq_ids) {
    std::vector<llama_token> input_ids(n_decode_batch_, 0);
    for (size_t i = 0; i < seq_ids.size(); i++) {
        int batch_id = seq_ids[i];
        // 单batch模式下所有序列共用batch 0
        if (n_decode_batch_ == 1)
            batch_id = 0;
        input_ids[batch_id] = batches[i];
    }
    auto input_data = decode_input_map_.at(decode_model_->GetInputName(0));
    try {
        if (embed_npu_ && embedding_layer_->is_npu_ready()) {
            embedding_layer_->get_embedding_batch_npu(
                input_ids.data(), input_ids.size(), input_data.Data());
        } else {
            embedding_layer_->get_embedding_batch(input_ids, input_data.Data());
        }
    } catch (...) {
        LLAMA_LOG_ERROR("%s: failed to get embedding batch\n", __func__);
        return -1;
    }
    decode_model_->SetInput(decode_model_->GetInputName(0), input_data);
    return 0;
}

int HouMoLLModel::Impl::set_xlenght_for_decode(std::vector<int> &seq_ids) {
    // 0 表示该物理 batch 未参与本轮 decode；有效 valid_length 至少为 1。
    std::vector<int32_t> valid_length_vec(n_decode_batch_, 0);
    std::vector<int32_t> current_length_vec(n_decode_batch_, 0);
    for (int seq_id : seq_ids) {
        const int batch_id = n_decode_batch_ == 1 ? 0 : seq_id;
        if (batch_id < 0 || batch_id >= n_decode_batch_) {
            LLAMA_LOG_ERROR("%s: invalid seq_id=%d mapped batch_id=%d, "
                            "n_decode_batch_=%d\n",
                            __func__, seq_id, batch_id, n_decode_batch_);
            return -1;
        }
        const int32_t valid_length =
            decode_valid_length_[seq_id] > 0 ? decode_valid_length_[seq_id] : 1;
        if (valid_length >= context_length_) {
            LLAMA_LOG_WARN("%s: valid_length seq_id=%d batch_id=%d value=%d "
                           ">= context_length_, end of decoding\n",
                           __func__, seq_id, batch_id, valid_length);
            return 1;
        }
        valid_length_vec[batch_id] = valid_length;
        current_length_vec[batch_id] = 1;
        LLAMA_LOG_DEBUG("%s: seq_id=%d -> batch_id=%d valid_length=%d "
                        "current_length=1\n",
                        __func__, seq_id, batch_id, valid_length);
    }
    for (size_t i = 0; i < valid_length_vec.size(); i++) {
        if (valid_length_vec[i] >= context_length_) {
            LLAMA_LOG_WARN("%s: valid_length_vec[%zu] = %d >= context_length_, "
                           "end of decoding\n",
                           __func__, i, valid_length_vec[i]);
            return 1;
        }
    }
    {
        // 工具链编译成多batch模型,特征如下
        // current_length, current_length_batch1, ...current_length_batchx
        // valid_length, valid_length_batch1, ...valid_length_batchx
        if (decode_current_length_names_.size() != (size_t)n_decode_batch_ ||
            decode_valid_length_names_.size() != (size_t)n_decode_batch_) {
            LLAMA_LOG_ERROR("%s: decode_current_length_names_.size() = %zu != "
                            "n_decode_batch_ = %d\n",
                            __func__, decode_current_length_names_.size(),
                            n_decode_batch_);
            return -1;
        }

        for (size_t i = 0; i < valid_length_vec.size(); i++) {
            tcim::Status status;
            std::string name;
            // 未参与本轮 decode 的 batch 不能覆盖 valid_length，否则可能
            // 使用错误位置推进并破坏该 slot 的 KV cache。
            if (valid_length_vec[i] > 0) {
                name = decode_valid_length_names_[i];
                auto decode_valid_length = decode_input_map_.at(name);
                set_input_data(decode_valid_length, 1, valid_length_vec[i]);
                status = decode_model_->SetInput(name, decode_valid_length);
                if (status != tcim::Status::OK) {
                    LLAMA_LOG_ERROR("%s: failed to set input %s\n", __func__,
                                    name.c_str());
                    return -1;
                }
            }
            // current_length 设置为0， 期望工具链不再计算。
            // http://jira.houmo.ai/browse/HMSW-4468
            name = decode_current_length_names_[i];
            auto current_valid_length = decode_input_map_.at(name);
            set_input_data(current_valid_length, 1, current_length_vec[i]);
            // LLAMA_LOG_DEBUG("name = %s, current_valid_length: %d\n",
            //                 name.c_str(), 1);
            status = decode_model_->SetInput(name, current_valid_length);
            if (status != tcim::Status::OK) {
                LLAMA_LOG_ERROR("%s: failed to set input %s\n", __func__,
                                name.c_str());
                return -1;
            }
        }
    }
    return 0;
}
void HouMoLLModel::Impl::lora_init(std::string file_name, void *adapter) {
    struct lora_item item;
    item.adapter = adapter;
    item.scale = 0.0;
    lora_map[file_name] = item;
    LLAMA_LOG_INFO("lora FILE_name IS %s\n", file_name.c_str());
}
void HouMoLLModel::Impl::lora_set_scale(void *adapter, float scale) {
    for (auto iter = lora_map.begin(); iter != lora_map.end(); ++iter) {
        if (iter->second.adapter == adapter) {
            iter->second.scale = scale;
        }
    }
}
void HouMoLLModel::Impl::lora_clear() {
    // scale set 0
    for (auto iter = lora_map.begin(); iter != lora_map.end(); ++iter) {
        iter->second.scale = 0;
    }
}

// HouMoLLModel constructor and destructor
HouMoLLModel::HouMoLLModel() = default;

HouMoLLModel::~HouMoLLModel() {
    for (auto *impl : impls_mtp_) delete impl;
    for (auto *impl : impls_) delete impl;
}

bool HouMoLLModel::route_seq(int global_seq_id, bool is_mtp,
                             SeqRoute &route) const {
    const auto &impls = is_mtp ? impls_mtp_ : impls_;
    const auto &group_ids = is_mtp ? mtp_group_ids_ : group_ids_;
    const auto &local_limits = is_mtp ? local_seq_max_mtp_ : local_seq_max_;
    const bool initialized = is_mtp ? mtp_initialized_ : target_initialized_;
    const int global_limit = is_mtp ? mtp_seq_max_ : target_seq_max_;

    if (!initialized || impls.empty() || global_seq_id < 0 ||
        global_seq_id >= global_limit || local_limits.size() != impls.size() ||
        group_ids.size() != impls.size()) {
        return false;
    }

    const int group_count = static_cast<int>(impls.size());
    const int group_index = global_seq_id % group_count;
    const int local_seq_id = global_seq_id / group_count;
    if (impls[group_index] == nullptr || local_seq_id < 0 ||
        local_seq_id >= local_limits[group_index]) {
        return false;
    }

    route = {group_index, group_ids[group_index], local_seq_id,
             impls[group_index]};
    return true;
}

void HouMoLLModel::state_write(llama_io_write_i &io, llama_seq_id seq_id,
                               llama_state_seq_flags flags, bool is_mtp) const {
    if (seq_id < 0) {
        throw std::runtime_error("Houmo full-context state is not supported; save a sequence instead");
    }
    SeqRoute route;
    if (!route_seq(seq_id, is_mtp, route)) {
        throw std::runtime_error("invalid Houmo sequence id");
    }
    const houmo_state_header header = {
        HOUMO_STATE_MAGIC,
        HOUMO_STATE_VERSION,
        flags,
        seq_id,
        route.group_id,
        route.local_seq_id,
    };
    io.write(&header, sizeof(header));
    route.impl->state_write(io, route.local_seq_id, flags);
}

void HouMoLLModel::state_read(llama_io_read_i &io, llama_seq_id seq_id,
                              llama_state_seq_flags flags, bool is_mtp) {
    if (seq_id < 0) {
        throw std::runtime_error("Houmo full-context state is not supported; restore a sequence instead");
    }
    SeqRoute route;
    if (!route_seq(seq_id, is_mtp, route)) {
        throw std::runtime_error("invalid Houmo sequence id");
    }
    houmo_state_header header = {};
    io.read(&header, sizeof(header));
    if (header.magic != HOUMO_STATE_MAGIC) {
        throw std::runtime_error("wrong Houmo state magic");
    }
    if (header.version != HOUMO_STATE_VERSION) {
        throw std::runtime_error("unsupported Houmo state version");
    }
    if (header.flags != flags) {
        throw std::runtime_error("Houmo state flags mismatch");
    }
    // Source and destination slots may be routed to different physical model
    // groups. The model payload validates tensor names, shapes, strides and
    // dtypes against the destination group before committing any data.
    route.impl->state_read(io, route.local_seq_id, flags);
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        const llama_pos restored_pos = static_cast<llama_pos>(route.impl->state_position(route.local_seq_id));
        if (restored_pos < 0) {
            throw std::runtime_error("invalid restored Houmo sequence position");
        }
        memory_->seq_rm(seq_id, 0, -1);
        memory_->seq_add(seq_id, 0, 0, restored_pos);
    }
}

// Forward method calls to the implementation
bool HouMoLLModel::houmo_load(llama_model_loader &ml, std::vector<int> device_ids) {
    if (!impls_.empty() || !impls_mtp_.empty()) {
        LLAMA_LOG_ERROR("%s: model is already loaded\n", __func__);
        return false;
    }
    LLAMA_LOG_INFO("%s: model arch name: %s\n", __func__, ml.arch_name.c_str());
    auto fail_load = [this]() {
        for (auto *impl : impls_mtp_) delete impl;
        for (auto *impl : impls_) delete impl;
        impls_mtp_.clear();
        impls_.clear();
        mtp_group_ids_.clear();
        group_ids_.clear();
        local_seq_max_mtp_.clear();
        local_seq_max_.clear();
        memories_mtp_.clear();
        memories_.clear();
        mtp_seq_max_ = 0;
        target_seq_max_ = 0;
        mtp_initialized_ = false;
        target_initialized_ = false;
        return false;
    };
    const bool embedding_model = ml.arch_name.find("bert") != std::string::npos;
    group_ids_ = embedding_model ? std::vector<int>{0} : collect_hmm_groups(ml, false);
    if (group_ids_.empty()) {
        LLAMA_LOG_ERROR("%s: no complete target HMM group found\n", __func__);
        return false;
    }
    tcim::Module::WeightManager shared_wm;
    std::shared_ptr<HoumoEmbeddingLayer> shared_embedding;
    for (int group_id : group_ids_) {
        Impl *impl = create_impl(ml.arch_name, false);
        if (!impl) return fail_load();
        impl->set_load_group_id(group_id);
        if (shared_wm.GetInitStatus() == tcim::Status::OK) impl->set_weight_manager(shared_wm);
        if (shared_embedding) impl->set_embedding_layer(shared_embedding);
        if (!impl->houmo_load(ml, device_ids)) {
            delete impl;
            return fail_load();
        }
        if (impls_.empty()) {
            shared_wm = impl->weight_manager();
            shared_embedding = impl->embedding_layer();
        }
        impls_.push_back(impl);
        LLAMA_LOG_INFO("%s: loaded target HMM group %d\n", __func__, group_id);
    }

    if (gguf_has_mtp_tensors(ml)) {
        mtp_group_ids_ = collect_hmm_groups(ml, true);
        if (mtp_group_ids_.size() != 1 || mtp_group_ids_.front() != 0) {
            LLAMA_LOG_ERROR("%s: only MTP group 0 is supported\n", __func__);
            return fail_load();
        }
        Impl *impl = create_impl(ml.arch_name, true, impls_.front());
        if (!impl) return fail_load();
        impl->set_load_group_id(0);
        impl->set_weight_manager(shared_wm);
        impl->set_embedding_layer(shared_embedding);
        if (!impl->houmo_load(ml, device_ids)) {
            delete impl;
            return fail_load();
        }
        impls_mtp_.push_back(impl);
    }
    return true;
}

bool HouMoLLModel::houmo_init(houmo_memory_i *memory, int seq_max,
                              ggml_abort_callback abort_callback,
                              void *abort_callback_data,
                              bool is_mtp) {
    if (memory_ == nullptr) {
        memory_ = memory;
    } else if (memory_ != memory) {
        LLAMA_LOG_ERROR("%s: target and draft contexts must share Houmo memory\n", __func__);
        return false;
    }

    if (is_mtp) {
        if (mtp_initialized_) {
            LLAMA_LOG_ERROR("%s: MTP runtime is already initialized\n", __func__);
            return false;
        }
        if (impls_mtp_.empty()) {
            LLAMA_LOG_ERROR("%s: requested MTP ctx but no MTP impl loaded\n", __func__);
            return false;
        }
        if (seq_max != 1 || impls_mtp_.size() != 1) {
            LLAMA_LOG_ERROR("%s: MTP only supports np=1 and one HMM group, got seq_max=%d group_count=%zu\n",
                            __func__, seq_max, impls_mtp_.size());
            return false;
        }
        std::string reason;
        if (!impls_mtp_.front()->supports_seq_max(1, reason)) {
            LLAMA_LOG_ERROR("%s: MTP group 0 cannot initialize seq_max=1: %s\n",
                            __func__, reason.c_str());
            return false;
        }
        memory->bind(this, true);
        memories_mtp_.push_back(std::make_unique<GroupMemory>(memory, 0, 1));
        impls_mtp_.front()->houmo_init(memories_mtp_.back().get(), 1,
                                       abort_callback, abort_callback_data);
        local_seq_max_mtp_ = {1};
        mtp_seq_max_ = 1;
        mtp_initialized_ = true;
    } else {
        if (target_initialized_) {
            LLAMA_LOG_ERROR("%s: target runtime is already initialized\n", __func__);
            return false;
        }
        const int count = static_cast<int>(impls_.size());
        if (count == 0 || seq_max < count) {
            LLAMA_LOG_ERROR("%s: seq_max=%d is smaller than group_count=%d\n",
                            __func__, seq_max, count);
            return false;
        }
        std::vector<int> new_local_seq_max;
        new_local_seq_max.reserve(count);
        for (int i = 0; i < count; ++i) {
            const int local_seq_max = (seq_max + count - 1 - i) / count;
            std::string reason;
            if (!impls_[i]->supports_seq_max(local_seq_max, reason)) {
                LLAMA_LOG_ERROR("%s: target group %d cannot initialize local_seq_max=%d: %s\n",
                                __func__, group_ids_[i], local_seq_max,
                                reason.c_str());
                return false;
            }
            new_local_seq_max.push_back(local_seq_max);
        }
        for (int i = 0; i < count; ++i) {
            if (i == 0) {
                memory->bind(this, false);
            }
            memories_.push_back(std::make_unique<GroupMemory>(memory, i, count));
            impls_[i]->houmo_init(memories_.back().get(), new_local_seq_max[i],
                                  abort_callback, abort_callback_data);
        }
        local_seq_max_ = std::move(new_local_seq_max);
        target_seq_max_ = seq_max;
        target_initialized_ = true;
    }
    return true;
}

int HouMoLLModel::houmo_prefill(std::vector<llama_token> &tokens, int seq_id,
                                float *logits) {
    SeqRoute route;
    if (!route_seq(seq_id, false, route)) {
        LLAMA_LOG_ERROR("%s: invalid target seq_id=%d (seq_max=%d)\n",
                        __func__, seq_id, target_seq_max_);
        return -1;
    }
    return route.impl->houmo_prefill(tokens, route.local_seq_id, logits);
}

int HouMoLLModel::houmo_decode(std::vector<llama_token> &batches,
                               std::vector<int> &seq_ids,
                               std::vector<float *> logits) {
    if (!target_initialized_ || impls_.empty() ||
        batches.size() != seq_ids.size() || logits.size() != seq_ids.size()) {
        LLAMA_LOG_ERROR("%s: target runtime is not initialized or input sizes do not match\n", __func__);
        return -1;
    }
    std::map<int, std::vector<size_t>> groups;
    std::vector<SeqRoute> routes(seq_ids.size());
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        if (!route_seq(seq_ids[i], false, routes[i])) {
            LLAMA_LOG_ERROR("%s: invalid target seq_id=%d (seq_max=%d)\n",
                            __func__, seq_ids[i], target_seq_max_);
            return -1;
        }
        groups[routes[i].group_index].push_back(i);
    }
    for (const auto &[group, indices] : groups) {
        std::vector<llama_token> local_tokens;
        std::vector<int> local_ids;
        std::vector<float *> local_logits;
        for (size_t i : indices) {
            local_tokens.push_back(batches[i]);
            local_ids.push_back(routes[i].local_seq_id);
            local_logits.push_back(logits[i]);
        }
        int ret = routes[indices.front()].impl->houmo_decode(
            local_tokens, local_ids, local_logits);
        if (ret != 0) return ret;
    }
    return 0;
}

int HouMoLLModel::houmo_asr_force_align(std::vector<llama_token> &tokens, int seq_id,
                                        std::vector<float *> &logits_out) {
    SeqRoute route;
    if (!route_seq(seq_id, false, route)) {
        LLAMA_LOG_ERROR("%s: invalid target seq_id=%d (seq_max=%d)\n",
                        __func__, seq_id, target_seq_max_);
        return -1;
    }
    return route.impl->houmo_asr_force_align(tokens, route.local_seq_id,
                                             logits_out);
}

int HouMoLLModel::houmo_verify(std::vector<llama_token> &tokens, int seq_id,
                               std::vector<float *> &logits_out) {
    SeqRoute route;
    if (!route_seq(seq_id, false, route)) {
        LLAMA_LOG_ERROR("%s: invalid target seq_id=%d (seq_max=%d)\n",
                        __func__, seq_id, target_seq_max_);
        return -1;
    }
    return route.impl->houmo_verify(tokens, route.local_seq_id, logits_out);
}

int HouMoLLModel::houmo_verify_accept(int seq_id, uint16_t n_accepted) {
    SeqRoute route;
    if (!route_seq(seq_id, false, route)) {
        LLAMA_LOG_ERROR("%s: invalid target seq_id=%d (seq_max=%d)\n",
                        __func__, seq_id, target_seq_max_);
        return -1;
    }
    return route.impl->houmo_verify_accept(route.local_seq_id, n_accepted);
}

int HouMoLLModel::n_decode_batch() {
    if (impls_.empty()) return 0;
    int result = impls_.front()->n_decode_batch();
    for (auto *impl : impls_) result = std::min(result, impl->n_decode_batch());
    return result;
}

int HouMoLLModel::n_prefill_batch() {
    if (impls_.empty()) return 0;
    int result = impls_.front()->n_prefill_batch();
    for (auto *impl : impls_) result = std::min(result, impl->n_prefill_batch());
    return result;
}

uint32_t HouMoLLModel::n_context_length() {
    if (impls_.empty()) return 0;
    uint32_t result = impls_.front()->n_context_length();
    for (auto *impl : impls_) result = std::max(result, impl->n_context_length());
    return result;
}

uint32_t HouMoLLModel::n_context_length(int seq_id) const {
    SeqRoute route;
    return route_seq(seq_id, false, route) ? route.impl->n_context_length() : 0;
}

int HouMoLLModel::houmo_prefill(const float *embeddings, int n_tokens,
                                int seq_id, float *logits) {
    SeqRoute route;
    if (!route_seq(seq_id, false, route)) {
        LLAMA_LOG_ERROR("%s: invalid target seq_id=%d (seq_max=%d)\n",
                        __func__, seq_id, target_seq_max_);
        return -1;
    }
    return route.impl->houmo_prefill(embeddings, n_tokens,
                                     route.local_seq_id, logits);
}

int HouMoLLModel::houmo_embedding(
    const std::map<int32_t, std::vector<llama_token>> &batches,
    std::map<int, float *> &embeddings) {
    if (!target_initialized_ || impls_.empty()) return -1;
    return impls_.front()->houmo_embedding(batches, embeddings);
}
bool HouMoLLModel::is_embedding() { return !impls_.empty() && impls_.front()->is_embedding(); }
bool HouMoLLModel::is_asr_force_aligner() { return !impls_.empty() && impls_.front()->is_asr_force_aligner(); }
bool HouMoLLModel::has_decode_model() {
    return !impls_.empty() &&
           std::all_of(impls_.begin(), impls_.end(), [](Impl *impl) { return impl->has_decode_model(); });
}
int HouMoLLModel::max_parallel_sequences() const {
    return impls_.empty() ? 0 : impls_.front()->max_parallel_sequences();
}

// ---- MTP forwarders ----
bool HouMoLLModel::is_mtp_model() const {
    return !impls_.empty() && impls_.front()->is_mtp_model();
}

bool HouMoLLModel::has_mtp_modules() const {
    return !impls_mtp_.empty();
}

int HouMoLLModel::houmo_decode_mtp(const float *h_in,
                                   const std::vector<llama_token> &tokens,
                                   int seq_id,
                                   std::vector<float *> &logits_out,
                                   float *h_out,
                                   int32_t start_pos) {
    SeqRoute route;
    if (!route_seq(seq_id, true, route)) {
        LLAMA_LOG_ERROR("%s: invalid MTP seq_id=%d (seq_max=%d)\n",
                        __func__, seq_id, mtp_seq_max_);
        return -1;
    }
    return route.impl->houmo_decode_mtp(h_in, tokens, route.local_seq_id,
        logits_out, h_out, start_pos);
}

void HouMoLLModel::set_pre_norm_hidden_dst(float *dst, int n_rows) {
    for (auto *impl : impls_) impl->set_pre_norm_hidden_dst(dst, n_rows);
}

void HouMoLLModel::lora_init(std::string file_name, void *adapter) {
    for (auto *impl : impls_) impl->lora_init(file_name, adapter);
}
void HouMoLLModel::lora_set_scale(void *adapter, float scale) {
    for (auto *impl : impls_) impl->lora_set_scale(adapter, scale);
}
void HouMoLLModel::lora_clear() {
    for (auto *impl : impls_) impl->lora_clear();
}
