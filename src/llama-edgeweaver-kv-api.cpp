#include "llama.h"

#ifdef LLAMAEDGE_ENABLE_KV_LAYER_EXPORT

#include "llama-kv-cache.h"
#include "llama-context.h"

uint32_t llamaedge_kv_cell_count(struct llama_context * ctx, llama_seq_id seq_id) {
    if (!ctx) {
        return 0;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return 0;
    }
    return kv->cell_count_for_seq(seq_id);
}

bool llamaedge_kv_layer_export_meta(
        struct llama_context * ctx,
                int32_t        layer_id,
                int32_t      * out_k_type,
                uint64_t     * out_k_row_size,
                int32_t      * out_v_type,
                uint64_t     * out_v_row_size_or_elem_size,
                uint32_t     * out_n_embd_v_gqa,
                bool         * out_v_trans) {
    if (!ctx) {
        return false;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return false;
    }
    return kv->layer_export_meta(
            layer_id,
            out_k_type,
            out_k_row_size,
            out_v_type,
            out_v_row_size_or_elem_size,
            out_n_embd_v_gqa,
            out_v_trans);
}

size_t llamaedge_kv_layer_export_k(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
                int32_t        layer_id,
                uint8_t      * dst,
                size_t         dst_size) {
    if (!ctx || (!dst && dst_size > 0)) {
        return 0;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return 0;
    }
    return kv->layer_export_k(seq_id, layer_id, dst, dst_size);
}

size_t llamaedge_kv_layer_export_v(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
                int32_t        layer_id,
                uint8_t      * dst,
                size_t         dst_size) {
    if (!ctx || (!dst && dst_size > 0)) {
        return 0;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return 0;
    }
    return kv->layer_export_v(seq_id, layer_id, dst, dst_size);
}

#endif // LLAMAEDGE_ENABLE_KV_LAYER_EXPORT

#ifdef LLAMAEDGE_ENABLE_KV_LAYER_IMPORT

#include "llama-kv-cache.h"
#include "llama-context.h"

bool llamaedge_kv_layer_import_prepare(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
           const llama_pos   * pos,
                uint32_t       cell_count) {
    if (!ctx || (cell_count > 0 && !pos)) {
        return false;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return false;
    }
    return kv->layer_import_prepare(seq_id, pos, cell_count);
}

size_t llamaedge_kv_layer_import_k(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
                int32_t        layer_id,
           const uint8_t    * src,
                size_t         src_size) {
    if (!ctx || (src_size > 0 && !src)) {
        return 0;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return 0;
    }
    return kv->layer_import_k(seq_id, layer_id, src, src_size);
}

size_t llamaedge_kv_layer_import_v(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
                int32_t        layer_id,
           const uint8_t    * src,
                size_t         src_size) {
    if (!ctx || (src_size > 0 && !src)) {
        return 0;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return 0;
    }
    return kv->layer_import_v(seq_id, layer_id, src, src_size);
}

bool llamaedge_kv_layer_import_prepare_append(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
           const llama_pos   * pos,
                uint32_t       cell_count) {
    if (!ctx || (cell_count > 0 && !pos)) {
        return false;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return false;
    }
    return kv->layer_import_prepare_append(seq_id, pos, cell_count);
}

size_t llamaedge_kv_layer_import_k_range(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
                int32_t        layer_id,
           const llama_pos   * pos,
                uint32_t       cell_count,
           const uint8_t    * src,
                size_t         src_size) {
    if (!ctx || (cell_count > 0 && (!pos || !src))) {
        return 0;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return 0;
    }
    return kv->layer_import_k_range(seq_id, layer_id, pos, cell_count, src, src_size);
}

size_t llamaedge_kv_layer_import_v_range(
        struct llama_context * ctx,
                llama_seq_id   seq_id,
                int32_t        layer_id,
           const llama_pos   * pos,
                uint32_t       cell_count,
           const uint8_t    * src,
                size_t         src_size) {
    if (!ctx || (cell_count > 0 && (!pos || !src))) {
        return 0;
    }
    auto * memory = ctx->get_memory();
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        return 0;
    }
    return kv->layer_import_v_range(seq_id, layer_id, pos, cell_count, src, src_size);
}

#endif // LLAMAEDGE_ENABLE_KV_LAYER_IMPORT
