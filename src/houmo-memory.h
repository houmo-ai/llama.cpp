#pragma once
#include "llama-memory.h"

/**
 * @brief A memory interface for Houmo
 * 当前不支持管理，空实现防止应用调用失败。
 */
class houmo_memory_i : public llama_memory_i {
  public:
    houmo_memory_i() = default;
    virtual ~houmo_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit
    // into the cache return a context object containing the ubatches and memory
    // state required to process them check the
    // llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(llama_batch_allocr &balloc,
                                                uint32_t n_ubatch,
                                                bool embd_all) override {
        (void) balloc;
        (void) n_ubatch;
        (void) embd_all;
        return nullptr;
    }

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() override { return nullptr; }

    // prepare for any pending memory updates, such as shifts, defrags, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context *lctx,
                                                 bool optimize) override {
        (void) lctx;
        (void) optimize;
        return nullptr;
    }

    // getters
    virtual bool get_can_shift() const override { return true; }

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the
    // metadata
    virtual void clear(bool data) override {
        (void) data;
        return;
    }

    virtual bool seq_rm(llama_seq_id seq_id, llama_pos p0,
                        llama_pos p1) override {
        (void) seq_id;
        (void) p0;
        (void) p1;
        return true;
    }
    virtual void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
                        llama_pos p0, llama_pos p1) override {
        (void) seq_id_src;
        (void) seq_id_dst;
        (void) p0;
        (void) p1;
        return;
    }
    virtual void seq_keep(llama_seq_id seq_id) override {
        (void) seq_id;
        return;
    }
    virtual void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                         llama_pos shift) override {
        (void) seq_id;
        (void) p0;
        (void) p1;
        (void) shift;
        return;
    }
    virtual void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                         int d) override {
        (void) seq_id;
        (void) p0;
        (void) p1;
        (void) d;
        return;
    }

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const override {
        (void) seq_id;
        return 0;
    }
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const override {
        (void) seq_id;
        return -1;
    }

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i &io,
                             llama_seq_id seq_id = -1) const override {
        (void) io;
        (void) seq_id;
        return;
    }
    virtual void state_read(llama_io_read_i &io,
                            llama_seq_id seq_id = -1) override {
        (void) io;
        (void) seq_id;
        return;
    }
};
