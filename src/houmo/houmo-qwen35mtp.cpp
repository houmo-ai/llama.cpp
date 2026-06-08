#include "houmo-qwen35mtp.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include "ggml.h"
#include "llama-impl.h"
#include "time-perf.h"

namespace {

// Last dim of a tensor shape (e.g. n_embd or n_vocab).
size_t last_dim(const tcim::TensorInfo &info) {
    const auto shape = info.Shape();
    return shape.empty() ? 0 : static_cast<size_t>(shape.back());
}

// Second-last (= seq) dim of a typical [B, S, H] tensor.
// Falls back to 1 if shape rank < 2.
size_t seq_dim(const tcim::TensorInfo &info) {
    const auto shape = info.Shape();
    if (shape.size() < 2) {
        return 1;
    }
    return static_cast<size_t>(shape[shape.size() - 2]);
}

// Same as above but the source is already a packed 2-byte-per-element buffer.
void set_padded_raw16_input(tcim::Module *m, const std::string &name,
                            tcim::Tensor &host_tensor, const void *src_raw,
                            size_t valid_rows, size_t row_elems) {
    const size_t cap_bytes = host_tensor.MemSize();
    const size_t valid_bytes = valid_rows * row_elems * 2;
    if (valid_bytes > cap_bytes) {
        LLAMA_LOG_ERROR("set_padded_raw16_input: '%s' valid=%zu > cap=%zu",
                        name.c_str(), valid_bytes, cap_bytes);
        return;
    }
    std::vector<uint8_t> buf(cap_bytes, 0);
    if (src_raw != nullptr && valid_bytes > 0) {
        std::memcpy(buf.data(), src_raw, valid_bytes);
    }
    host_tensor.Buffer().CopyFromHost(buf.data(), cap_bytes);
    m->SetInput(name, host_tensor);
}

// Copy raw fp16 rows from a float* carrier buffer whose logical row stride is
// still sizeof(float) * row_elems. Only the low 2 bytes per logical element are
// considered valid payload.
void set_padded_raw16_input_from_float_carrier(tcim::Module *m,
                                               const std::string &name,
                                               tcim::Tensor &host_tensor,
                                               const float *src_carrier,
                                               size_t valid_rows,
                                               size_t row_elems) {
    const size_t cap_bytes = host_tensor.MemSize();
    const size_t row_bytes = row_elems * sizeof(ggml_fp16_t);
    const size_t valid_bytes = valid_rows * row_bytes;
    const size_t src_row_stride = row_elems * sizeof(float);
    if (valid_bytes > cap_bytes) {
        LLAMA_LOG_ERROR("set_padded_raw16_input_from_float_carrier: '%s' valid=%zu > cap=%zu",
                        name.c_str(), valid_bytes, cap_bytes);
        return;
    }
    std::vector<uint8_t> buf(cap_bytes, 0);
    if (src_carrier != nullptr && valid_rows > 0 && row_elems > 0) {
        const auto *src = reinterpret_cast<const uint8_t *>(src_carrier);
        for (size_t r = 0; r < valid_rows; ++r) {
            std::memcpy(buf.data() + r * row_bytes,
                        src + r * src_row_stride,
                        row_bytes);
        }
    }
    host_tensor.Buffer().CopyFromHost(buf.data(), cap_bytes);
    m->SetInput(name, host_tensor);
}

// Zero-init a module input tensor by allocating a host buffer of zeros and
// pushing it via SetInput.
void zero_input_via_setinput(tcim::Module *model, const std::string &name) {
    if (model == nullptr || name.empty()) {
        return;
    }
    auto info = model->GetInputInfo(name).AsContiguous();
    auto host = tcim::Tensor::CreateHostTensor(info);
    std::vector<uint8_t> zeros(host.MemSize(), 0);
    host.Buffer().CopyFromHost(zeros.data(), host.MemSize());
    model->SetInput(name, host);
}

// Set a scalar i32 input by filling the entire tensor with the value.
// Matches demo_mtp.py's _set_scalar_i32 behavior.
void set_scalar_i32(tcim::Module *m, const std::string &name,
                    tcim::Tensor &host_tensor, int32_t value) {
    const size_t n_elems = host_tensor.MemSize() / sizeof(int32_t);
    std::vector<int32_t> buf(n_elems, value);
    host_tensor.Buffer().CopyFromHost(buf.data(), n_elems * sizeof(int32_t));
    m->SetInput(name, host_tensor);
}

} // namespace

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

void HoumoQwen35MTP::houmo_init(houmo_memory_i *memory, int seq_max,
                                ggml_abort_callback abort_callback,
                                void *abort_callback_data) {
    abort_callback_data_ = abort_callback_data;
    abort_callback_ = abort_callback;
    model_inout_init(seq_max);
    if (is_embedding()) {
        memory_ = nullptr;
    } else {
        memory_ = memory;
    }
    init_state_cache(seq_max);

    if (!is_mtp_model()) {
        return;
    }

    // MTP-specific initialization below.
    if (prefill_model_) {
        init_mtp_model_names_fixed(prefill_model_.get(), &mtp_state_.mtp_prefill);
        zero_input_via_setinput(prefill_model_.get(),
                                mtp_state_.mtp_prefill.past_key_cache_in);
        zero_input_via_setinput(prefill_model_.get(),
                                mtp_state_.mtp_prefill.past_value_cache_in);
    }
    if (decode_model_) {
        init_mtp_model_names_fixed(decode_model_.get(), &mtp_state_.mtp_decode);
    }

    // Device-link cache tensors between prefill/decode modules.
    if (prefill_model_ && decode_model_) {
        const auto &pns = mtp_state_.mtp_prefill;
        const auto &dns = mtp_state_.mtp_decode;

        if (!pns.past_key_cache_in.empty() && !dns.past_key_cache_in.empty()) {
            auto pt = prefill_model_->GetDevInput(pns.past_key_cache_in);
            if (pt.GetInitStatus() == tcim::Status::OK) {
                decode_model_->SetDevInput(dns.past_key_cache_in, pt);
                LLAMA_LOG_INFO("%s: linked MTP kcache prefill->decode\n", __func__);
            }
        }
        if (!pns.past_value_cache_in.empty() && !dns.past_value_cache_in.empty()) {
            auto pt = prefill_model_->GetDevInput(pns.past_value_cache_in);
            if (pt.GetInitStatus() == tcim::Status::OK) {
                decode_model_->SetDevInput(dns.past_value_cache_in, pt);
                LLAMA_LOG_INFO("%s: linked MTP vcache prefill->decode\n", __func__);
            }
        }
    }
}

void HoumoQwen35MTP::init_mtp_model_names_fixed(tcim::Module *model,
                                                HoumoQwen35MtpNameSet *out) {
    if (model == nullptr || out == nullptr) {
        return;
    }
    if (model->GetInputNum() < 6 || model->GetOutputNum() < 2) {
        LLAMA_LOG_ERROR("%s: unexpected MTP module IO counts in=%zu out=%zu\n",
                        __func__, model->GetInputNum(), model->GetOutputNum());
        return;
    }

    out->input_embedding_in  = model->GetInputName(0);
    out->hidden_in           = model->GetInputName(1);
    out->past_seq_length     = model->GetInputName(2);
    out->current_input_length = model->GetInputName(3);
    out->past_key_cache_in   = model->GetInputName(4);
    out->past_value_cache_in = model->GetInputName(5);
    // Collect position_ids from remaining inputs (index >= 6).
    // Matches demo_mtp.py MtpNames.position_ids.
    out->position_ids.clear();
    for (size_t idx = 6; idx < model->GetInputNum(); ++idx) {
        const std::string name = model->GetInputName(idx);
        if (name.find("position_ids") != std::string::npos) {
            out->position_ids.push_back(name);
        }
    }
    out->mtp_logits_out = model->GetOutputName(0);
    out->mtp_hidden_out = model->GetOutputName(1);

    LLAMA_LOG_DEBUG(
        "%s: fixed MTP IO hidden_in='%s' embed_in='%s' valid_len='%s' "
        "cur_len='%s' kcache='%s' vcache='%s' pos_ids#=%zu logits='%s' hidden_out='%s'\n",
        __func__,
        out->hidden_in.c_str(),
        out->input_embedding_in.c_str(),
        out->past_seq_length.c_str(),
        out->current_input_length.c_str(),
        out->past_key_cache_in.c_str(),
        out->past_value_cache_in.c_str(),
        out->position_ids.size(),
        out->mtp_logits_out.c_str(),
        out->mtp_hidden_out.c_str());
}

// ---------------------------------------------------------------------------
// clear_mtp_device_cache — debug fallback to zero TCIM cache after rollback
// ---------------------------------------------------------------------------

void HoumoQwen35MTP::clear_mtp_device_cache(int seq_id) {
    (void) seq_id;
    LLAMA_LOG_WARN("%s: MTP TCIM cache rollback detected, zeroing cache "
                   "on prefill and decode modules\n", __func__);

    // Zero on prefill module
    if (prefill_model_) {
        const auto &pns = mtp_state_.mtp_prefill;
        if (!pns.past_key_cache_in.empty()) {
            zero_input_via_setinput(prefill_model_.get(), pns.past_key_cache_in);
        }
        if (!pns.past_value_cache_in.empty()) {
            zero_input_via_setinput(prefill_model_.get(), pns.past_value_cache_in);
        }
    }

    // Also zero on decode module (even though cache is device-linked,
    // explicitly zeroing both ensures correctness regardless of link semantics).
    if (decode_model_) {
        const auto &dns = mtp_state_.mtp_decode;
        if (!dns.past_key_cache_in.empty()) {
            zero_input_via_setinput(decode_model_.get(), dns.past_key_cache_in);
        }
        if (!dns.past_value_cache_in.empty()) {
            zero_input_via_setinput(decode_model_.get(), dns.past_value_cache_in);
        }
    }
}

// ---------------------------------------------------------------------------
// houmo_decode_mtp — public dispatch (validates, then routes to prefill/decode)
// ---------------------------------------------------------------------------

int HoumoQwen35MTP::houmo_decode_mtp(const float *h_in,
                                     const std::vector<llama_token> &tokens,
                                     int seq_id,
                                     std::vector<float *> &logits_out,
                                     float *h_out,
                                     int32_t start_pos) {
    if (!is_mtp_model()) {
        LLAMA_LOG_ERROR("%s: called on non-MTP gguf\n", __func__);
        return -1;
    }
    if (tokens.empty()) {
        return 0;
    }
    if (tokens.size() != logits_out.size()) {
        LLAMA_LOG_ERROR("%s: tokens.size()=%zu != logits_out.size()=%zu\n",
                        __func__, tokens.size(), logits_out.size());
        return -1;
    }
    if (h_in == nullptr) {
        LLAMA_LOG_ERROR("%s: h_in is null\n", __func__);
        return -1;
    }
    if (embedding_layer_ == nullptr) {
        LLAMA_LOG_ERROR("%s: embedding_layer_ is null\n", __func__);
        return -1;
    }

    // ----- Dispatch to separate prefill / decode paths.
    const int n_tokens = static_cast<int>(tokens.size());
    
    const bool use_decode = (n_tokens == 1) && (decode_model_ != nullptr) &&
                            !mtp_state_.mtp_decode.hidden_in.empty();

    if (use_decode) {
        return run_mtp_decode(h_in, tokens, seq_id, logits_out, h_out, start_pos);
    } else {
        return run_mtp_prefill(h_in, tokens, seq_id, logits_out, h_out, start_pos);
    }
}

// ---------------------------------------------------------------------------
// run_mtp_prefill — multi-token MTP forward (uses prefill_model_)
// ---------------------------------------------------------------------------

int HoumoQwen35MTP::run_mtp_prefill(const float *h_in,
                                    const std::vector<llama_token> &tokens,
                                    int seq_id,
                                    std::vector<float *> &logits_out,
                                    float *h_out,
                                    int32_t start_pos) {
    const int n_tokens = static_cast<int>(tokens.size());
    tcim::Module *m = prefill_model_.get();
    auto &in_map = prefill_input_map_;
    auto &out_map = prefill_output_map_;
    const auto &ns = mtp_state_.mtp_prefill;

    LLAMA_LOG_INFO("[MTP-PREFILL] seq_id=%d n_tokens=%d start_pos=%d "
                   "memory=%s\n",
                   seq_id, n_tokens, (int)start_pos,
                   memory_ ? "yes" : "no");

    if (m == nullptr) {
        LLAMA_LOG_ERROR("[MTP-PREFILL] prefill_model_ is null\n");
        return -1;
    }
    if (ns.hidden_in.empty() || ns.input_embedding_in.empty() ||
        ns.mtp_logits_out.empty() || ns.mtp_hidden_out.empty()) {
        LLAMA_LOG_ERROR("[MTP-PREFILL] MTP names unresolved\n");
        return -1;
    }

    // Look up host tensors
    auto hidden_it = in_map.find(ns.hidden_in);
    if (hidden_it == in_map.end()) {
        LLAMA_LOG_ERROR("[MTP-PREFILL] hidden tensor '%s' not in input map\n",
                        ns.hidden_in.c_str());
        return -1;
    }
    auto &hidden_tensor = hidden_it->second;

    auto embed_it = in_map.find(ns.input_embedding_in);
    if (embed_it == in_map.end()) {
        LLAMA_LOG_ERROR("[MTP-PREFILL] embedding tensor '%s' not in input map\n",
                        ns.input_embedding_in.c_str());
        return -1;
    }
    auto &embed_tensor = embed_it->second;

    const size_t hidden_row = last_dim(hidden_tensor.Info());
    const size_t embed_row  = last_dim(embed_tensor.Info());
    const size_t hidden_seq = seq_dim(hidden_tensor.Info());
    const size_t embed_seq  = seq_dim(embed_tensor.Info());

    const size_t chunk_cap = std::min<size_t>({
        static_cast<size_t>(prefill_length_), hidden_seq, embed_seq});
    if (chunk_cap == 0) {
        LLAMA_LOG_ERROR("[MTP-PREFILL] invalid chunk_cap=0 (prefill_length_=%d "
                        "hidden_seq=%zu embed_seq=%zu)\n",
                        prefill_length_, hidden_seq, embed_seq);
        return -1;
    }

    // Validate tokens
    const uint32_t embed_vocab = embedding_layer_->get_vocab_size();
    for (int i = 0; i < n_tokens; ++i) {
        const auto tok = tokens[static_cast<size_t>(i)];
        if (tok < 0 || static_cast<uint32_t>(tok) >= embed_vocab) {
            LLAMA_LOG_ERROR("[MTP-PREFILL] invalid token id %d at idx=%d "
                            "(embed_vocab=%u)\n",
                            (int)tok, i, embed_vocab);
            return -1;
        }
    }

    // Compute past_seq_len
    int past_seq_len = 0;
    if (start_pos >= 0) {
        past_seq_len = std::max(0, static_cast<int>(start_pos) - 1);
    } else if (memory_) {
        past_seq_len = memory_->seq_pos_min(seq_id);
    } else {
        past_seq_len = mtp_state_.mtp_past_seq_len[seq_id];
    }
    LLAMA_LOG_DEBUG("[MTP-PREFILL] prefill past_seq_len=%d", past_seq_len);

    // The caller (common/speculative.cpp) already constructs demo_mtp.py-style
    // pairs: hidden = h_{i-1}, token = token_i. Therefore this backend must feed
    // h_in[0..n-1] with tokens[0..n-1] as-is. Do NOT shift again here.
    size_t cursor = 0;
    while (cursor < static_cast<size_t>(n_tokens)) {
        const size_t mtp_valid = std::min(chunk_cap,
                                          static_cast<size_t>(n_tokens) - cursor);
        const int chunk_past_seq_len =
            past_seq_len + static_cast<int>(cursor);
        LLAMA_LOG_DEBUG("prefill chunk cursor=%zu mtp_valid=%zu past_seq_len=%d",
                  cursor, mtp_valid, chunk_past_seq_len);

        // 1. Set hidden input: h_in uses float* only as carrier; payload is raw fp16.
        const float *chunk_hidden = h_in + cursor * hidden_row;
        set_padded_raw16_input_from_float_carrier(
            m, ns.hidden_in, hidden_tensor, chunk_hidden, mtp_valid, hidden_row);

        // 2. Set embedding input: tokens[cursor..cursor+mtp_valid-1]
        std::vector<uint8_t> embed_src(mtp_valid * embed_row * 2, 0);
        embedding_layer_->get_embedding_batch(tokens.data() + cursor,
                                              mtp_valid,
                                              embed_src.data());
        set_padded_raw16_input(m, ns.input_embedding_in, embed_tensor,
                               embed_src.data(), mtp_valid, embed_row);

        // 3. Scalars - use set_scalar_i32 to fill entire tensor (matches demo_mtp.py)
        if (!ns.past_seq_length.empty()) {
            auto it = in_map.find(ns.past_seq_length);
            if (it != in_map.end()) {
                set_scalar_i32(m, ns.past_seq_length, it->second,
                               chunk_past_seq_len);
            }
        }
        if (!ns.current_input_length.empty()) {
            auto it = in_map.find(ns.current_input_length);
            if (it != in_map.end()) {
                set_scalar_i32(m, ns.current_input_length, it->second,
                               static_cast<int32_t>(mtp_valid));
            }
        }

        // 3b. Set position_ids (matches demo_mtp.py _set_position_ids).
        // Build continuous range [chunk_past_seq_len, chunk_past_seq_len + seq_cap).
        if (!ns.position_ids.empty()) {
            const size_t seq_cap = seq_dim(hidden_tensor.Info());
            std::vector<int32_t> pos_vec(static_cast<size_t>(seq_cap), 0);
            for (size_t i = 0; i < seq_cap; ++i) {
                pos_vec[i] = chunk_past_seq_len + static_cast<int32_t>(i);
            }
            for (const auto &pn : ns.position_ids) {
                auto it = in_map.find(pn);
                if (it != in_map.end()) {
                    auto &t = it->second;
                    size_t cap_elems = t.MemSize() / sizeof(int32_t);
                    std::vector<int32_t> buf(cap_elems, 0);
                    size_t n = std::min(cap_elems, pos_vec.size());
                    for (size_t i = 0; i < n; ++i) {
                        buf[i] = pos_vec[i];
                    }
                    t.Buffer().CopyFromHost(buf.data(),
                                            cap_elems * sizeof(int32_t));
                    m->SetInput(pn, t);
                }
            }
        }

        // 4. Run + Sync
        auto status = m->Run();
        if (status != tcim::Status::OK) {
            LLAMA_LOG_ERROR("[MTP-PREFILL] Run failed status=%d at cursor=%zu "
                            "mtp_valid=%zu\n",
                            (int)status, cursor, mtp_valid);
            return -1;
        }
        m->Sync();

        // 5. Read logits (mtp_valid entries, matching the pre-aligned input pairs)
        auto dev = m->GetOutput(ns.mtp_logits_out);
        auto out_it = out_map.find(ns.mtp_logits_out);
        if (out_it == out_map.end()) {
            LLAMA_LOG_ERROR("[MTP-PREFILL] missing logits output '%s'\n",
                            ns.mtp_logits_out.c_str());
            return -1;
        }
        const auto info = dev.Info();
        if (info.DataType() != tcim::DataType::FLOAT16) {
            LLAMA_LOG_ERROR("[MTP-PREFILL] unexpected logits output dtype=%d\n",
                            (int)info.DataType());
            return -1;
        }
        auto *base = static_cast<const ggml_fp16_t *>(dev.Buffer().Data());
        const size_t row = last_dim(info);
        for (size_t i = 0; i < mtp_valid; ++i) {
            if (logits_out[cursor + i] == nullptr) {
                continue;
            }
            ggml_fp16_to_fp32_row(base + i * row, logits_out[cursor + i],
                                  static_cast<int64_t>(row));
        }
        cursor += mtp_valid;
    }

    // 6. Set h_out to the last row of h_in (main model's hidden for the
    //    last token). This matches demo_mtp.py where last_hidden for the
    //    first MTP decode step comes from the main model, not MTP output.
    if (h_out != nullptr) {
        const float *last_hidden = h_in + static_cast<size_t>(n_tokens - 1) * hidden_row;
        std::memcpy(h_out, last_hidden, hidden_row * sizeof(ggml_fp16_t));
    }

    // 7. Advance MTP memory and cache tracker (mtp_valid = n_tokens)
    mtp_state_.mtp_past_seq_len[seq_id] = past_seq_len + n_tokens;
    mtp_state_.mtp_cache_end[seq_id] = past_seq_len + n_tokens;
    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, n_tokens);
        LLAMA_LOG_DEBUG("[MTP-PREFILL] seq_id=%d n_tokens=%d mtp_valid=%d "
                       "past_seq_len=%d -> cache_end=%d pos_max=%d\n",
                       seq_id, n_tokens, n_tokens, past_seq_len,
                       (int)mtp_state_.mtp_cache_end[seq_id],
                       (int)memory_->seq_pos_max(seq_id));
    }
    return 0;
}

// ---------------------------------------------------------------------------
// run_mtp_decode — single-token MTP forward (uses decode_model_)
// ---------------------------------------------------------------------------

int HoumoQwen35MTP::run_mtp_decode(const float *h_in,
                                   const std::vector<llama_token> &tokens,
                                   int seq_id,
                                   std::vector<float *> &logits_out,
                                   float *h_out,
                                   int32_t start_pos) {
    tcim::Module *m = decode_model_.get();
    auto &in_map = decode_input_map_;
    auto &out_map = decode_output_map_;
    const auto &ns = mtp_state_.mtp_decode;

    LLAMA_LOG_DEBUG("[MTP-DECODE] seq_id=%d start_pos=%d memory=%s\n",
                   seq_id, (int)start_pos, memory_ ? "yes" : "no");

    if (m == nullptr) {
        LLAMA_LOG_ERROR("[MTP-DECODE] decode_model_ is null\n");
        return -1;
    }
    if (ns.hidden_in.empty() || ns.input_embedding_in.empty() ||
        ns.mtp_logits_out.empty() || ns.mtp_hidden_out.empty()) {
        LLAMA_LOG_ERROR("[MTP-DECODE] MTP names unresolved\n");
        return -1;
    }

    // Look up host tensors
    auto hidden_it = in_map.find(ns.hidden_in);
    if (hidden_it == in_map.end()) {
        LLAMA_LOG_ERROR("[MTP-DECODE] hidden tensor '%s' not in decode input "
                        "map (map size=%zu)\n",
                        ns.hidden_in.c_str(), in_map.size());
        return -1;
    }
    auto &hidden_tensor = hidden_it->second;

    auto embed_it = in_map.find(ns.input_embedding_in);
    if (embed_it == in_map.end()) {
        LLAMA_LOG_ERROR("[MTP-DECODE] embedding tensor '%s' not in decode "
                        "input map\n",
                        ns.input_embedding_in.c_str());
        return -1;
    }
    auto &embed_tensor = embed_it->second;

    const size_t hidden_row = last_dim(hidden_tensor.Info());
    const size_t embed_row  = last_dim(embed_tensor.Info());

    // Validate token
    const uint32_t embed_vocab = embedding_layer_->get_vocab_size();
    const auto tok = tokens[0];
    if (tok < 0 || static_cast<uint32_t>(tok) >= embed_vocab) {
        LLAMA_LOG_ERROR("[MTP-DECODE] invalid token id %d (embed_vocab=%u)\n",
                        (int)tok, embed_vocab);
        return -1;
    }

    // Compute past_seq_len
    int past_seq_len = 0;
    if (start_pos >= 0) {
        past_seq_len = std::max(0, static_cast<int>(start_pos) - 1);
    } else if (memory_) {
        past_seq_len = memory_->seq_pos_min(seq_id);
    } else {
        past_seq_len = mtp_state_.mtp_past_seq_len[seq_id];
    }
    LLAMA_LOG_DEBUG("[MTP-DECODE] decode past_seq_len=%d", past_seq_len);
    PERF_START(mtp_set_inputs);
    // 1. Set hidden input (single row, raw fp16 carried through float*).
    set_padded_raw16_input_from_float_carrier(
        m, ns.hidden_in, hidden_tensor, h_in, 1, hidden_row);

    // 2. Set embedding input (single row)
    std::vector<uint8_t> embed_src(embed_row * 2, 0);
    embedding_layer_->get_embedding_batch(tokens.data(), 1, embed_src.data());
    set_padded_raw16_input(m, ns.input_embedding_in, embed_tensor,
                           embed_src.data(), 1, embed_row);

    // 3. Scalars - use set_scalar_i32 to fill entire tensor (matches demo_mtp.py)
    if (!ns.past_seq_length.empty()) {
        auto it = in_map.find(ns.past_seq_length);
        if (it != in_map.end()) {
            set_scalar_i32(m, ns.past_seq_length, it->second, past_seq_len);
        }
    }
    if (!ns.current_input_length.empty()) {
        auto it = in_map.find(ns.current_input_length);
        if (it != in_map.end()) {
            set_scalar_i32(m, ns.current_input_length, it->second, 1);
        }
    }

    // 3b. Set position_ids (matches demo_mtp.py _set_position_ids, length=1).
    if (!ns.position_ids.empty()) {
        std::vector<int32_t> pos_vec(1, past_seq_len);
        for (const auto &pn : ns.position_ids) {
            auto it = in_map.find(pn);
            if (it != in_map.end()) {
                auto &t = it->second;
                size_t cap_elems = t.MemSize() / sizeof(int32_t);
                std::vector<int32_t> buf(cap_elems, 0);
                size_t n = std::min(cap_elems, pos_vec.size());
                for (size_t i = 0; i < n; ++i) buf[i] = pos_vec[i];
                t.Buffer().CopyFromHost(buf.data(),
                                        cap_elems * sizeof(int32_t));
                m->SetInput(pn, t);
            }
        }
    }
    PERF_STOP(mtp_set_inputs);
    PERF_START(mtp_run_once);
    // 4. Run + Sync
    auto status = m->Run();
    if (status != tcim::Status::OK) {
        LLAMA_LOG_ERROR("[MTP-DECODE] Run failed status=%d\n", (int)status);
        return -1;
    }
    m->Sync();
    PERF_STOP(mtp_run_once);
    PERF_START(mtp_read_outputs);
    // 5. Read logits
    {
        auto dev = m->GetOutput(ns.mtp_logits_out);
        const auto info = dev.Info();
        if (info.DataType() != tcim::DataType::FLOAT16) {
            LLAMA_LOG_ERROR("[MTP-DECODE] unexpected logits output dtype=%d\n",
                            (int)info.DataType());
            return -1;
        }
        if (logits_out[0] != nullptr) {
            auto *base = static_cast<const ggml_fp16_t *>(dev.Buffer().Data());
            const size_t row = last_dim(info);
            ggml_fp16_to_fp32_row(base, logits_out[0], static_cast<int64_t>(row));
        }
    }

    // 6. Read hidden output
    if (h_out != nullptr) {
        auto dev = m->GetOutput(ns.mtp_hidden_out);
        auto out_it = out_map.find(ns.mtp_hidden_out);
        if (out_it == out_map.end()) {
            LLAMA_LOG_ERROR("[MTP-DECODE] missing hidden output '%s'\n",
                            ns.mtp_hidden_out.c_str());
            return -1;
        }
        const auto info = dev.Info();
        const size_t hrow = last_dim(info);
        if (info.DataType() == tcim::DataType::FLOAT16) {
            const void *base = dev.Buffer().Data();
            std::memcpy(h_out, base, hrow * sizeof(ggml_fp16_t));
        } else {
            LLAMA_LOG_ERROR("[MTP-DECODE] unexpected hidden output dtype=%d\n",
                            (int)info.DataType());
            return -1;
        }
    }

    // 7. Advance MTP memory and cache tracker
    mtp_state_.mtp_past_seq_len[seq_id] = past_seq_len + 1;
    mtp_state_.mtp_cache_end[seq_id] = past_seq_len + 1;
    if (memory_) {
        memory_->seq_add(seq_id, 0, 0, 1);
        LLAMA_LOG_DEBUG("[MTP-DECODE] seq_id=%d start_pos=%d past_seq_len=%d "
                        "-> cache_end=%d pos_max=%d\n",
                        seq_id, (int)start_pos, past_seq_len,
                        (int)mtp_state_.mtp_cache_end[seq_id],
                        (int)memory_->seq_pos_max(seq_id));
    }
    PERF_STOP(mtp_read_outputs);
    return 0;
}
