#ifndef LLAMAEDGE_HOOKS_H
#define LLAMAEDGE_HOOKS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Keep this header self-contained. Most in-tree users include llama.h first,
// but EdgeWeaver smoke tests and external shims may include <llamaedge/hooks.h>
// directly. Mirror llama.h's export macro guard instead of requiring the full
// llama public header here.
#ifndef LLAMA_API
#    ifdef LLAMA_SHARED
#        if defined(_WIN32) && !defined(__MINGW32__)
#            ifdef LLAMA_BUILD
#                define LLAMA_API __declspec(dllexport)
#            else
#                define LLAMA_API __declspec(dllimport)
#            endif
#        else
#            define LLAMA_API __attribute__ ((visibility ("default")))
#        endif
#    else
#        define LLAMA_API
#    endif
#endif

// Forward declarations
struct llama_context;
struct llama_batch;

typedef int (*llamaedge_split_segment_callback)(
    void * user_data,
    uint32_t segment_index,
    uint32_t layer_begin,
    uint32_t layer_end,
    bool before_segment);

// Execute one decoder batch as contiguous transformer-layer segments. This
// prototype API is intentionally limited to architectures with validated
// split-graph builders; callers must provide strictly increasing layer ends
// whose final value is the model layer count.
LLAMA_API int llamaedge_decode_split_graph(
    struct llama_context * ctx,
    const struct llama_batch * batch,
    const uint32_t * layer_ends,
    uint32_t n_segments);
LLAMA_API int llamaedge_decode_split_graph_with_callback(
    struct llama_context * ctx,
    const struct llama_batch * batch,
    const uint32_t * layer_ends,
    uint32_t n_segments,
    llamaedge_split_segment_callback callback,
    void * user_data);

// ============================================================
// Hook infrastructure flags
// ============================================================
#define LLAMAEDGE_HOOK_PRE_PREFILL    (1 << 0)
#define LLAMAEDGE_HOOK_POST_PREFILL   (1 << 1)
#define LLAMAEDGE_HOOK_KV_EXPORT      (1 << 2)
#define LLAMAEDGE_HOOK_KV_INSTALL     (1 << 3)

// ============================================================
// Hook function types
// ============================================================
typedef int (*llamaedge_kv_export_fn)(
    struct llama_context * ctx, void * user_data,
    int32_t layer_id, int32_t seq_id,
    const float * k_data, const float * v_data,
    uint32_t n_tokens, uint32_t n_kv_heads,
    uint32_t n_embd_head_k, uint32_t n_embd_head_v);

typedef int (*llamaedge_kv_install_fn)(
    struct llama_context * ctx, void * user_data,
    int32_t layer_id, int32_t seq_id,
    const float * k_data, const float * v_data,
    uint32_t n_tokens, uint32_t n_kv_heads,
    uint32_t n_embd_head_k, uint32_t n_embd_head_v);

typedef int (*llamaedge_hook_fn)(struct llama_context * ctx, void * user_data);

// ============================================================
// Legacy hook types (B2 compatibility)
// ============================================================
typedef struct llamaedge_hook {
    uint32_t               hook_type;
    llamaedge_hook_fn      fn;
    void *                 user_data;
    struct llamaedge_hook * next;
} llamaedge_hook;

// ============================================================
// Chunk streaming types
// ============================================================
typedef struct llamaedge_chunk_config {
    uint32_t chunk_size_tokens;
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
} llamaedge_chunk_config;

typedef struct llamaedge_tx_desc {
    int32_t     session_id;
    int32_t     layer_id;
    int32_t     chunk_id;
    uint32_t    token_begin;
    uint32_t    token_count;
    uint32_t    n_kv_heads;
    uint32_t    n_embd_head_k;
    uint32_t    n_embd_head_v;
    size_t      payload_size;
    const uint8_t * payload;
    bool        in_use;
} llamaedge_tx_desc;

typedef struct llamaedge_frontier {
    int32_t * layer_frontier;
    int32_t   session_id;
    uint32_t  n_layers;
    void *    pending_spans;
} llamaedge_frontier;

// ============================================================
// TX ring types
// ============================================================
typedef struct llamaedge_tx_ring_config {
    uint32_t ring_size;
    uint32_t payload_pool_bytes;
    uint32_t n_destinations;
} llamaedge_tx_ring_config;

typedef struct llamaedge_tx_ring llamaedge_tx_ring;

// ============================================================
// Warm threshold types
// ============================================================
typedef struct llamaedge_warm_config {
    uint32_t n_warm_layers;
    uint32_t n_warm_chunks;
    uint32_t chunk_size_tokens;
} llamaedge_warm_config;

// ============================================================
// B4 enums and config types
// ============================================================
typedef enum llamaedge_thinning_policy {
    LLAMAEDGE_THIN_NONE = 0,
    LLAMAEDGE_THIN_L2_NORM = 1,
} llamaedge_thinning_policy;

typedef enum llamaedge_quant_bitplan {
    LLAMAEDGE_Q3_K4V3 = 0,
    LLAMAEDGE_Q4_K4V4 = 1,
    LLAMAEDGE_Q4_K4V3 = 2,
    LLAMAEDGE_Q3_K3V2 = 3,
} llamaedge_quant_bitplan;

typedef enum llamaedge_profile {
    LLAMAEDGE_PROFILE_CONSERVATIVE = 0,
    LLAMAEDGE_PROFILE_BALANCED = 1,
    LLAMAEDGE_PROFILE_AGGRESSIVE = 2,
} llamaedge_profile;

typedef struct llamaedge_thinning_config {
    bool        enabled;
    llamaedge_thinning_policy policy;
    uint32_t    top_k;
    float       top_ratio;
    uint32_t    protected_token_count;
} llamaedge_thinning_config;

typedef struct llamaedge_quant_config {
    bool        enabled;
    llamaedge_quant_bitplan bitplan;
    bool        use_wht;
} llamaedge_quant_config;

typedef struct llamaedge_b4_config {
    llamaedge_profile         profile;
    llamaedge_thinning_config thinning;
    llamaedge_quant_config    quant;
    llamaedge_warm_config     warm;
    uint32_t                  chunk_size_tokens;
} llamaedge_b4_config;

// ============================================================
// B4 pipeline (primary public API)
// ============================================================
typedef struct llamaedge_b4_pipeline llamaedge_b4_pipeline;

// ============================================================
// Wire format types
// ============================================================
typedef struct llamaedge_chunk_header {
    int32_t     chunk_id;
    int32_t     layer_id;
    uint32_t    token_begin;
    uint32_t    token_count;
    uint8_t     k_bits;
    uint8_t     v_bits;
    bool        use_wht;
    uint32_t    n_kv_heads;
    uint32_t    n_embd_head_k;
    uint32_t    n_embd_head_v;
} llamaedge_chunk_header;

typedef struct llamaedge_kv_layer_data {
    int32_t layer_id;
    int32_t  k_type;
    uint64_t k_size_row;
    uint32_t k_n_tokens;
    const uint8_t * k_data;
    int32_t  v_type;
    uint64_t v_size_row;
    uint32_t v_n_tokens;
    const uint8_t * v_data;
    bool     v_trans;
    uint32_t v_size_el;
    uint32_t n_embd_v_gqa;
    const uint8_t * v_data_trans;
} llamaedge_kv_layer_data;

typedef struct llamaedge_kv_parsed_state {
    uint32_t stream_count;
    uint32_t stream_index;
    uint32_t cell_count;
    bool v_trans;
    uint32_t n_layer;
    uint32_t n_tokens;
    size_t payload_offset;
    size_t payload_bytes;
    llamaedge_kv_layer_data * layers;
    const uint8_t * payload_end;
} llamaedge_kv_parsed_state;

// ============================================================
// Profile change callback
// ============================================================
typedef void (*llamaedge_profile_change_fn)(
    llamaedge_profile old_profile,
    llamaedge_profile new_profile,
    void * user_data);

// ============================================================
// TX Ring API
// ============================================================
LLAMA_API llamaedge_tx_ring * llamaedge_tx_ring_create(const llamaedge_tx_ring_config * config);
LLAMA_API void llamaedge_tx_ring_destroy(llamaedge_tx_ring * ring);
LLAMA_API int llamaedge_tx_ring_push(llamaedge_tx_ring * ring, uint32_t destination, const llamaedge_tx_desc * desc);
LLAMA_API int llamaedge_tx_ring_pop(llamaedge_tx_ring * ring, uint32_t destination, llamaedge_tx_desc * desc);
LLAMA_API uint32_t llamaedge_tx_ring_pending(const llamaedge_tx_ring * ring, uint32_t destination);
LLAMA_API uint8_t * llamaedge_payload_pool_alloc(llamaedge_tx_ring * ring, size_t size);
LLAMA_API void llamaedge_payload_pool_free(llamaedge_tx_ring * ring, uint8_t * payload);

// ============================================================
// KV export/install API
// ============================================================
LLAMA_API int llamaedge_kv_export_register(struct llama_context * ctx, llamaedge_kv_export_fn fn, void * user_data);
LLAMA_API int llamaedge_kv_export_unregister(struct llama_context * ctx, llamaedge_kv_export_fn fn);
LLAMA_API int llamaedge_kv_install_register(struct llama_context * ctx, llamaedge_kv_install_fn fn, void * user_data);
LLAMA_API int llamaedge_kv_install_unregister(struct llama_context * ctx, llamaedge_kv_install_fn fn);
LLAMA_API int llamaedge_kv_export_layers(struct llama_context * ctx, int32_t seq_id, uint32_t n_tokens);
LLAMA_API void llamaedge_kv_cell_diag(struct llama_context * ctx, int32_t seq_id);
LLAMA_API int llamaedge_kv_install_chunk(struct llama_context * ctx, int32_t seq_id, int32_t layer_id,
    const float * k_data, const float * v_data,
    uint32_t n_tokens, uint32_t n_kv_heads, uint32_t n_embd_head_k, uint32_t n_embd_head_v);

// ============================================================
// Frontier / warm threshold API
// ============================================================
LLAMA_API llamaedge_frontier * llamaedge_frontier_create(int32_t session_id, uint32_t n_layers);
LLAMA_API void llamaedge_frontier_destroy(llamaedge_frontier * frontier);
LLAMA_API void llamaedge_frontier_update(llamaedge_frontier * frontier, int32_t layer_id, uint32_t token_begin, uint32_t token_count);
// Safe for ordinary llama_decode only when the contiguous token prefix is
// present for every layer. Partial-layer warm checks are diagnostics/custom
// graph research paths outside this compatibility ABI.
LLAMA_API bool llamaedge_warm_threshold_met(const llamaedge_frontier * frontier, int32_t current_decode_pos, const llamaedge_warm_config * config);
LLAMA_API int32_t llamaedge_frontier_min(const llamaedge_frontier * frontier);

// ============================================================
// Thinning API
// ============================================================
LLAMA_API uint32_t llamaedge_thinning_score_and_select(const float * token_embeddings, uint32_t n_tokens,
    uint32_t n_embd, const llamaedge_thinning_config * config, uint32_t * selected_indices);
LLAMA_API void llamaedge_thinning_l2_scores(const float * token_embeddings, uint32_t n_tokens,
    uint32_t n_embd, float * scores_out);

// ============================================================
// Quantization API
// ============================================================
LLAMA_API size_t llamaedge_quantize_k(const float * k_data, uint32_t n_tokens, uint32_t n_kv_heads,
    uint32_t n_embd_head_k, const llamaedge_quant_config * config, uint8_t * quantized_out, float * scales_out);
LLAMA_API void llamaedge_dequantize_k(const uint8_t * quantized, const float * scales, uint32_t n_tokens,
    uint32_t n_kv_heads, uint32_t n_embd_head_k, const llamaedge_quant_config * config, float * k_data_out);
LLAMA_API size_t llamaedge_quantize_v(const float * v_data, uint32_t n_tokens, uint32_t n_kv_heads,
    uint32_t n_embd_head_v, const llamaedge_quant_config * config, uint8_t * quantized_out, float * scales_out);
LLAMA_API void llamaedge_dequantize_v(const uint8_t * quantized, const float * scales, uint32_t n_tokens,
    uint32_t n_kv_heads, uint32_t n_embd_head_v, const llamaedge_quant_config * config, float * v_data_out);

// ============================================================
// Profile / control plane API
// ============================================================
LLAMA_API int llamaedge_profile_change_register(struct llama_context * ctx, llamaedge_profile_change_fn fn, void * user_data);
LLAMA_API int llamaedge_profile_change_unregister(struct llama_context * ctx, llamaedge_profile_change_fn fn);
LLAMA_API int llamaedge_profile_apply(struct llama_context * ctx, llamaedge_profile profile);
LLAMA_API llamaedge_profile llamaedge_profile_get(const struct llama_context * ctx);

// ============================================================
// B4 pipeline API (primary)
// ============================================================
LLAMA_API llamaedge_b4_pipeline * llamaedge_b4_pipeline_create(const llamaedge_b4_config * config);
LLAMA_API void llamaedge_b4_pipeline_destroy(llamaedge_b4_pipeline * pipeline);
LLAMA_API llamaedge_profile llamaedge_b4_pipeline_get_profile(const llamaedge_b4_pipeline * pipeline);
LLAMA_API int llamaedge_b4_pipeline_update_profile(llamaedge_b4_pipeline * pipeline, llamaedge_profile profile);
LLAMA_API const llamaedge_thinning_config * llamaedge_b4_pipeline_get_thinning_config(const llamaedge_b4_pipeline * pipeline);
LLAMA_API const llamaedge_quant_config * llamaedge_b4_pipeline_get_quant_config(const llamaedge_b4_pipeline * pipeline);
LLAMA_API const llamaedge_warm_config * llamaedge_b4_pipeline_get_warm_config(const llamaedge_b4_pipeline * pipeline);
LLAMA_API uint32_t llamaedge_b4_pipeline_prefill(llamaedge_b4_pipeline * pipeline, struct llama_context * ctx,
    int32_t seq_id, uint32_t n_tokens, const float * embeddings, llamaedge_tx_ring * tx_ring);
LLAMA_API int llamaedge_b4_pipeline_install_chunk(llamaedge_b4_pipeline * pipeline, struct llama_context * ctx,
    int32_t seq_id, const uint8_t * quantized_k, const float * k_scales,
    const uint8_t * quantized_v, const float * v_scales,
    uint32_t n_tokens, uint32_t n_kv_heads, uint32_t n_embd_head_k, uint32_t n_embd_head_v);

// ============================================================
// Wire format API
// ============================================================
LLAMA_API llamaedge_kv_parsed_state * llamaedge_kv_parse_state(const uint8_t * serialized, size_t size);
LLAMA_API void llamaedge_kv_parse_state_free(llamaedge_kv_parsed_state * state);
LLAMA_API size_t llamaedge_b4_serialize_chunk_header(const llamaedge_chunk_header * hdr, uint8_t * dst, size_t dst_size);
LLAMA_API bool llamaedge_b4_parse_chunk_header(const uint8_t * src, size_t src_size, llamaedge_chunk_header * hdr_out);
LLAMA_API size_t llamaedge_state_get_size(struct llama_context * ctx, int32_t seq_id);
LLAMA_API size_t llamaedge_state_get_data(struct llama_context * ctx, uint8_t * dst, size_t size, int32_t seq_id);
LLAMA_API size_t llamaedge_state_set_data(struct llama_context * ctx, const uint8_t * src, size_t size, int32_t seq_id);

// ============================================================
// Hook management API (B2 compatibility)
// ============================================================
LLAMA_API void llamaedge_hook_init(struct llama_context * ctx);
LLAMA_API void llamaedge_hook_cleanup(struct llama_context * ctx);
LLAMA_API int llamaedge_hook_register(struct llama_context * ctx, const llamaedge_hook * hook);
LLAMA_API int llamaedge_hook_unregister(struct llama_context * ctx, const llamaedge_hook * hook);
LLAMA_API bool llamaedge_hook_is_active(struct llama_context * ctx, uint32_t hook_type);

#ifdef __cplusplus
}
#endif

#endif // LLAMAEDGE_HOOKS_H
