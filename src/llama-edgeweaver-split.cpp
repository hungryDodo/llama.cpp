#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama.h"

#include <vector>

#ifdef LLAMAEDGE_ENABLE_SPLIT_GRAPH
int llama_context::decode_split(
        const llama_batch & batch_inp,
        const uint32_t * layer_ends,
        uint32_t n_segments,
        llama_split_segment_callback callback,
        void * callback_user_data) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (!memory || !layer_ends || n_segments == 0) {
        LLAMA_LOG_ERROR("%s: decoder memory and at least one segment are required\n", __func__);
        return -1;
    }

    if (model.arch != LLM_ARCH_LLAMA && model.arch != LLM_ARCH_QWEN3) {
        LLAMA_LOG_ERROR("%s: architecture %s has no validated split-graph builder\n",
                __func__, model.arch_name().c_str());
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;
    const uint32_t n_layer = hparams.n_layer;

    uint32_t layer_begin = 0;
    for (uint32_t i = 0; i < n_segments; ++i) {
        if (layer_ends[i] <= layer_begin || layer_ends[i] > n_layer) {
            LLAMA_LOG_ERROR("%s: invalid segment %u layer range [%u, %u) for %u layers\n",
                    __func__, i, layer_begin, layer_ends[i], n_layer);
            return -1;
        }
        layer_begin = layer_ends[i];
    }
    if (layer_begin != n_layer) {
        LLAMA_LOG_ERROR("%s: final segment ends at layer %u, expected %u\n",
                __func__, layer_begin, n_layer);
        return -1;
    }

    if (batch_inp.n_tokens == 0 || static_cast<uint32_t>(batch_inp.n_tokens) > cparams.n_ubatch) {
        LLAMA_LOG_ERROR("%s: split-graph decode requires 0 < n_tokens <= n_ubatch (%d > %u)\n",
                __func__, batch_inp.n_tokens, cparams.n_ubatch);
        return -1;
    }

    if (cparams.embeddings || !sampling.samplers.empty()) {
        LLAMA_LOG_ERROR("%s: split-graph decode does not support embeddings or backend samplers\n", __func__);
        return -1;
    }

    const int64_t n_embd = hparams.n_embd_inp();
    if (n_embd != hparams.n_embd || hparams.f_embedding_scale != 0.0f) {
        LLAMA_LOG_ERROR("%s: unsupported embedding geometry or input scaling\n", __func__);
        return -1;
    }

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;
    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, false)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();
    memory_update(false);

    bool did_optimize = false;
    llama_memory_context_ptr mctx;
    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, false);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n",
                        __func__, mctx->get_status());
                return -2;
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                if (!did_optimize) {
                    did_optimize = true;
                    if (memory_update(true)) {
                        continue;
                    }
                }
                LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %u\n",
                        __func__, n_tokens_all);
                return 1;
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %u\n",
                        __func__, n_tokens_all);
                return -2;
        }
        break;
    }

    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n",
                __func__, n_outputs_all);
        return -2;
    }

    const llama_ubatch & ubatch = mctx->get_ubatch();
    if (ubatch.n_tokens != n_tokens_all) {
        LLAMA_LOG_ERROR("%s: memory module split the batch (%u of %u tokens)\n",
                __func__, ubatch.n_tokens, n_tokens_all);
        return -1;
    }

    if (n_outputs_all == n_tokens_all) {
        n_outputs = ubatch.n_tokens;
    } else {
        n_outputs = 0;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            n_outputs += static_cast<int32_t>(ubatch.output[i] != 0);
        }
    }

    std::vector<float> hidden(static_cast<size_t>(n_embd) * n_tokens_all);
    llama_ubatch segment_ubatch = ubatch;
    layer_begin = 0;

    for (uint32_t segment = 0; segment < n_segments; ++segment) {
        const uint32_t layer_end = layer_ends[segment];

        if (callback && callback(
                    callback_user_data, segment, layer_begin, layer_end, true) != 0) {
            LLAMA_LOG_ERROR("%s: segment %u start callback rejected execution\n", __func__, segment);
            return -4;
        }

        const auto finish_callback = [&]() {
            return !callback || callback(
                    callback_user_data, segment, layer_begin, layer_end, false) == 0;
        };

        if (segment > 0) {
            segment_ubatch.token = nullptr;
            segment_ubatch.embd = hidden.data();
        }

        ggml_status status;
        const auto * res = process_ubatch(
                segment_ubatch,
                LLM_GRAPH_TYPE_DECODER,
                mctx.get(),
                status,
                static_cast<int32_t>(layer_begin),
                static_cast<int32_t>(layer_end),
                segment == 0);

        if (!res) {
            finish_callback();
            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        if (layer_end < n_layer) {
            auto * t_hidden = res->get_embd();
            if (!t_hidden || ggml_nelements(t_hidden) != static_cast<int64_t>(hidden.size())) {
                LLAMA_LOG_ERROR("%s: segment %u returned invalid hidden-state geometry\n",
                        __func__, segment);
                finish_callback();
                return -3;
            }

            ggml_backend_t backend_hidden =
                    ggml_backend_sched_get_tensor_backend(sched.get(), t_hidden);
            GGML_ASSERT(backend_hidden != nullptr);

            // Graph execution is asynchronous on the backend stream. Queue the
            // D2H copy on that same stream, then wait before the next segment
            // consumes the host-side activation.
            ggml_backend_tensor_get_async(
                    backend_hidden,
                    t_hidden,
                    hidden.data(),
                    0,
                    hidden.size() * sizeof(float));
            ggml_backend_synchronize(backend_hidden);
        } else {
            auto * t_logits = res->get_logits();
            if (!t_logits || !logits.data || n_outputs <= 0) {
                LLAMA_LOG_ERROR("%s: final segment did not produce requested logits\n", __func__);
                finish_callback();
                return -3;
            }

            const int64_t n_vocab = vocab.n_tokens();
            GGML_ASSERT(static_cast<int64_t>(n_outputs) * n_vocab <= static_cast<int64_t>(logits.size));
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            ggml_backend_tensor_get_async(
                    backend_res,
                    t_logits,
                    logits.data,
                    0,
                    static_cast<size_t>(n_outputs) * n_vocab * sizeof(float));
            // A segment-finish callback may release EdgeWeaver's llama mutex.
            // Complete the final D2H transfer before that ownership handoff.
            if (callback) {
                ggml_backend_synchronize(backend_res);
            }
        }

        if (!finish_callback()) {
            LLAMA_LOG_ERROR("%s: segment %u finish callback failed\n", __func__, segment);
            return -4;
        }

        layer_begin = layer_end;
    }

    if (mctx->next()) {
        LLAMA_LOG_ERROR("%s: split-graph decode does not support a memory-split batch\n", __func__);
        return -1;
    }

    n_outputs = n_outputs_all;
    if (n_outputs > 0) {
        auto & out_ids = balloc->get_out_ids();
        if (out_ids.size() != static_cast<size_t>(n_outputs)) {
            LLAMA_LOG_ERROR("%s: invalid output mapping size\n", __func__);
            return -3;
        }
        for (int64_t i = 0; i < n_outputs; ++i) {
            output_ids[out_ids[i]] = i;
        }
    }

    return 0;
}

LLAMA_API int llamaedge_decode_split_graph(
        llama_context * ctx,
        const llama_batch * batch,
        const uint32_t * layer_ends,
        uint32_t n_segments) {
    if (!ctx || !batch || !layer_ends || n_segments == 0) {
        return -1;
    }

    const int ret = ctx->decode_split(*batch, layer_ends, n_segments);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode split graph, ret = %d\n", __func__, ret);
    }
    return ret;
}

LLAMA_API int llamaedge_decode_split_graph_with_callback(
        llama_context * ctx,
        const llama_batch * batch,
        const uint32_t * layer_ends,
        uint32_t n_segments,
        llamaedge_split_segment_callback callback,
        void * user_data) {
    if (!ctx || !batch || !layer_ends || n_segments == 0 || !callback) {
        return -1;
    }

    const int ret = ctx->decode_split(*batch, layer_ends, n_segments, callback, user_data);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode split graph, ret = %d\n", __func__, ret);
    }
    return ret;
}

#endif // LLAMAEDGE_ENABLE_SPLIT_GRAPH
