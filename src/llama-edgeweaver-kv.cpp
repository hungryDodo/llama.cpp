#include "llama-kv-cache.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

//
// EdgeWeaver per-layer KV helpers
//
#if defined(LLAMAEDGE_ENABLE_KV_LAYER_EXPORT) || defined(LLAMAEDGE_ENABLE_KV_LAYER_IMPORT)

// Collect cells for a sequence in logical position order rather than physical
// KV-cell index order. llama.cpp may reuse/free/shift cells under continuous
// batching, so physical cell order is not a stable proxy for token position.
static std::vector<uint32_t> llamaedge_seq_cells_by_pos(
        const llama_kv_cells & cells,
        llama_seq_id           seq_id) {
    std::vector<std::pair<llama_pos, uint32_t>> ordered;
    ordered.reserve(cells.size());
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.seq_has(i, seq_id)) {
            ordered.emplace_back(cells.pos_get(i), i);
        }
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });

    std::vector<uint32_t> out;
    out.reserve(ordered.size());
    for (const auto & item : ordered) {
        out.push_back(item.second);
    }
    return out;
}

static bool llamaedge_seq_cells_for_positions(
        const llama_kv_cells        & cells,
        llama_seq_id                  seq_id,
        const llama_pos             * pos,
        uint32_t                      cell_count,
        std::vector<uint32_t>       & out) {
    out.clear();
    if (cell_count == 0) {
        return true;
    }
    if (!pos) {
        LLAMA_LOG_ERROR("%s: null pos for non-empty import\n", __func__);
        return false;
    }

    out.reserve(cell_count);
    std::set<llama_pos> requested_positions;
    for (uint32_t pidx = 0; pidx < cell_count; ++pidx) {
        const llama_pos wanted = pos[pidx];
        if (!requested_positions.insert(wanted).second) {
            LLAMA_LOG_ERROR(
                "%s: duplicate requested position %d\n",
                __func__, static_cast<int>(wanted));
            out.clear();
            return false;
        }
        bool found = false;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.seq_has(i, seq_id) && cells.pos_get(i) == wanted) {
                out.push_back(i);
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

#endif // LLAMAEDGE_ENABLE_KV_LAYER_EXPORT || LLAMAEDGE_ENABLE_KV_LAYER_IMPORT

//
// per-layer KV export helper
//
#ifdef LLAMAEDGE_ENABLE_KV_LAYER_EXPORT

uint32_t llama_kv_cache::cell_count_for_seq(llama_seq_id seq_id) const {
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, (int) seq_id);
        return 0;
    }
    const auto & cells = v_cells[seq_to_stream[seq_id]];

    uint32_t count = 0;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.seq_has(i, seq_id)) {
            ++count;
        }
    }

    return count;
}

bool llama_kv_cache::layer_export_meta(
        int32_t   layer_id,
        int32_t * out_k_type,
        uint64_t * out_k_row_size,
        int32_t * out_v_type,
        uint64_t * out_v_row_size_or_elem_size,
        uint32_t * out_n_embd_v_gqa,
        bool     * out_v_trans) const {

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return false;
    }

    const auto & layer = layers[it->second];
    const uint32_t il = layer.il;

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    // K metadata
    if (out_k_type) {
        *out_k_type = (int32_t) layer.k->type;
    }
    if (out_k_row_size) {
        *out_k_row_size = ggml_row_size(layer.k->type, n_embd_k_gqa);
    }

    // V metadata
    if (out_v_trans) {
        *out_v_trans = v_trans;
    }
    if (out_n_embd_v_gqa) {
        *out_n_embd_v_gqa = n_embd_v_gqa;
    }

    if (layer.v) {
        if (out_v_type) {
            *out_v_type = (int32_t) layer.v->type;
        }
        if (out_v_row_size_or_elem_size) {
            if (v_trans) {
                *out_v_row_size_or_elem_size = ggml_type_size(layer.v->type);
            } else {
                *out_v_row_size_or_elem_size = ggml_row_size(layer.v->type, n_embd_v_gqa);
            }
        }
    } else {
        // MLA models may not have V cache
        if (out_v_type) {
            *out_v_type = -1;
        }
        if (out_v_row_size_or_elem_size) {
            *out_v_row_size_or_elem_size = 0;
        }
    }

    return true;
}

size_t llama_kv_cache::layer_export_k(
        llama_seq_id  seq_id,
        int32_t       layer_id,
        uint8_t     * dst,
        size_t        dst_size) const {

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, (int) seq_id);
        return 0;
    }
    if (!dst && dst_size > 0) {
        LLAMA_LOG_ERROR("%s: null destination buffer with non-zero size\n", __func__);
        return 0;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return 0;
    }

    const uint32_t strm = seq_to_stream[seq_id];
    const auto & cells  = v_cells[strm];
    const auto & layer  = layers[it->second];
    const uint32_t il   = layer.il;

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const size_t k_size_row = ggml_row_size(layer.k->type, n_embd_k_gqa);

    // Gather cells in logical token-position order, not physical cell order.
    const std::vector<uint32_t> cell_idxs = llamaedge_seq_cells_by_pos(cells, seq_id);

    const size_t required_size = cell_idxs.size() * k_size_row;
    if (dst_size < required_size || required_size == 0) {
        return required_size;
    }

    auto * k = layer.k_stream[strm];
    size_t offset = 0;
    for (const auto & idx : cell_idxs) {
        ggml_backend_tensor_get(k, dst + offset, idx * k_size_row, k_size_row);
        offset += k_size_row;
    }

    return offset;
}

size_t llama_kv_cache::layer_export_v(
        llama_seq_id  seq_id,
        int32_t       layer_id,
        uint8_t     * dst,
        size_t        dst_size) const {

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, (int) seq_id);
        return 0;
    }
    if (!dst && dst_size > 0) {
        LLAMA_LOG_ERROR("%s: null destination buffer with non-zero size\n", __func__);
        return 0;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        return 0;
    }

    const uint32_t strm = seq_to_stream[seq_id];
    const auto & cells  = v_cells[strm];
    const auto & layer  = layers[it->second];
    const uint32_t il   = layer.il;

    if (!layer.v) {
        return 0;
    }

    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    // Gather cells in logical token-position order, not physical cell order.
    const std::vector<uint32_t> cell_idxs = llamaedge_seq_cells_by_pos(cells, seq_id);

    size_t required_size = 0;
    size_t offset = 0;
    auto * v = layer.v_stream[strm];

    if (!v_trans) {
        const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
        required_size = cell_idxs.size() * v_size_row;

        if (dst_size < required_size || required_size == 0) {
            return required_size;
        }

        for (const auto & idx : cell_idxs) {
            ggml_backend_tensor_get(v, dst + offset, idx * v_size_row, v_size_row);
            offset += v_size_row;
        }
    } else {
        // Transposed V: each embedding element is stored as a row
        const size_t v_size_el = ggml_type_size(v->type);
        const uint32_t kv_size = cells.size();
        required_size = cell_idxs.size() * n_embd_v_gqa * v_size_el;

        if (dst_size < required_size || required_size == 0) {
            return required_size;
        }

        for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
            for (const auto & idx : cell_idxs) {
                const size_t src_offset = (idx + (size_t)j * kv_size) * v_size_el;
                ggml_backend_tensor_get(v, dst + offset, src_offset, v_size_el);
                offset += v_size_el;
            }
        }
    }

    return offset;
}

#endif // LLAMAEDGE_ENABLE_KV_LAYER_EXPORT

//
// per-layer KV import helper
//
#ifdef LLAMAEDGE_ENABLE_KV_LAYER_IMPORT

bool llama_kv_cache::layer_import_prepare(
        llama_seq_id      seq_id,
        const llama_pos * pos,
        uint32_t          cell_count) {

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, (int) seq_id);
        return false;
    }

    if (cell_count == 0) {
        return true;
    }
    if (!pos) {
        LLAMA_LOG_ERROR("%s: null pos for non-empty import\n", __func__);
        return false;
    }

    std::set<llama_pos> requested;
    for (uint32_t i = 0; i < cell_count; ++i) {
        if (!requested.insert(pos[i]).second) {
            LLAMA_LOG_ERROR("%s: duplicate requested position %d\n", __func__, (int) pos[i]);
            return false;
        }
    }

    const uint32_t strm = seq_to_stream[seq_id];
    auto & cells = v_cells[strm];

    // Phase 1 (preflight): count available free cells WITHOUT modifying state.
    // A cell counts as available if it is already empty or is exclusively
    // owned by this sequence. Shared cells remain occupied after seq_rm removes
    // only this sequence id, so counting them would make preflight succeed and
    // allow the destructive commit to fail after clearing the old sequence.
    uint32_t free_count = 0;
    for (uint32_t i = 0; i < cells.size() && free_count < cell_count; ++i) {
        if (cells.is_empty(i) ||
            (cells.seq_has(i, seq_id) && cells.seq_count(i) == 1)) {
            free_count++;
        }
    }

    if (free_count < cell_count) {
        LLAMA_LOG_ERROR("%s: not enough free cells (%u needed, %u available)\n",
                __func__, cell_count, free_count);
        return false;
    }

    // Phase 2 (commit): remove old cells for this seq_id, then allocate.
    seq_rm(seq_id, -1, -1);

    uint32_t found = 0;
    for (uint32_t i = 0; i < cells.size() && found < cell_count; ++i) {
        if (cells.is_empty(i)) {
            cells.pos_set(i, pos[found]);
            cells.seq_add(i, seq_id);
            found++;
        }
    }

    if (found != cell_count) {
        // Rollback: remove any cells we allocated so the caller can retry on a fresh context
        LLAMA_LOG_ERROR("%s: commit failed (%u needed, %u found), rolling back\n",
                __func__, cell_count, found);
        seq_rm(seq_id, -1, -1);
        return false;
    }

    return true;
}

bool llama_kv_cache::layer_import_prepare_append(
        llama_seq_id      seq_id,
        const llama_pos * pos,
        uint32_t          cell_count) {

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, (int) seq_id);
        return false;
    }
    if (cell_count == 0) {
        return true;
    }
    if (!pos) {
        LLAMA_LOG_ERROR("%s: null pos for non-empty import\n", __func__);
        return false;
    }

    const uint32_t strm = seq_to_stream[seq_id];
    auto & cells = v_cells[strm];

    std::set<llama_pos> present;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.seq_has(i, seq_id)) {
            present.insert(cells.pos_get(i));
        }
    }

    std::vector<llama_pos> missing;
    missing.reserve(cell_count);
    std::set<llama_pos> requested;
    for (uint32_t i = 0; i < cell_count; ++i) {
        if (!requested.insert(pos[i]).second) {
            LLAMA_LOG_ERROR("%s: duplicate requested position %d\n", __func__, (int) pos[i]);
            return false;
        }
        if (present.find(pos[i]) == present.end()) {
            missing.push_back(pos[i]);
        }
    }

    if (missing.empty()) {
        return true;
    }

    uint32_t free_count = 0;
    for (uint32_t i = 0; i < cells.size() && free_count < missing.size(); ++i) {
        if (cells.is_empty(i)) {
            free_count++;
        }
    }
    if (free_count < missing.size()) {
        LLAMA_LOG_ERROR("%s: not enough free cells (%zu needed, %u available)\n", __func__, missing.size(), free_count);
        return false;
    }

    uint32_t found = 0;
    for (uint32_t i = 0; i < cells.size() && found < missing.size(); ++i) {
        if (cells.is_empty(i)) {
            cells.pos_set(i, missing[found]);
            cells.seq_add(i, seq_id);
            found++;
        }
    }
    return found == missing.size();
}

size_t llama_kv_cache::layer_import_k(
        llama_seq_id      seq_id,
        int32_t           layer_id,
        const uint8_t   * src,
        size_t            src_size) {

    if (src_size > 0 && !src) {
        LLAMA_LOG_ERROR("%s: null source buffer for non-empty import\n", __func__);
        return 0;
    }
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, (int) seq_id);
        return 0;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("%s: invalid layer_id %d\n", __func__, layer_id);
        return 0;
    }

    const uint32_t strm  = seq_to_stream[seq_id];
    const auto & cells   = v_cells[strm];
    const auto & layer   = layers[it->second];
    const uint32_t il    = layer.il;

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const size_t k_size_row = ggml_row_size(layer.k->type, n_embd_k_gqa);

    // Gather cells in logical token-position order, so imported rows map to metadata/payload position order.
    const std::vector<uint32_t> cell_idxs = llamaedge_seq_cells_by_pos(cells, seq_id);

    const size_t expected_size = cell_idxs.size() * k_size_row;
    if (cell_idxs.empty() || src_size != expected_size) {
        LLAMA_LOG_ERROR("%s: size mismatch for layer %d (need %zu, got %zu)\n",
                __func__, layer_id, expected_size, src_size);
        return 0;
    }

    auto * k = layer.k_stream[strm];
    if (!k) {
        LLAMA_LOG_ERROR("%s: K tensor is null for layer %d\n", __func__, layer_id);
        return 0;
    }

    // Write K data: use single contiguous write when cells are sequential
    // (matches state_read_data fast path, critical for correct graph compilation)
    bool contiguous = true;
    for (size_t ci = 1; ci < cell_idxs.size(); ++ci) {
        if (cell_idxs[ci] != cell_idxs[ci-1] + 1) { contiguous = false; break; }
    }
    size_t offset = 0;
    if (contiguous && !cell_idxs.empty()) {
        ggml_backend_tensor_set(k, src, cell_idxs[0] * k_size_row, cell_idxs.size() * k_size_row);
        offset = cell_idxs.size() * k_size_row;
    } else {
        for (const auto & idx : cell_idxs) {
            ggml_backend_tensor_set(k, src + offset, idx * k_size_row, k_size_row);
            offset += k_size_row;
        }
    }

    return offset;
}

size_t llama_kv_cache::layer_import_v(
        llama_seq_id      seq_id,
        int32_t           layer_id,
        const uint8_t   * src,
        size_t            src_size) {

    if (src_size > 0 && !src) {
        LLAMA_LOG_ERROR("%s: null source buffer for non-empty import\n", __func__);
        return 0;
    }
    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, (int) seq_id);
        return 0;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("%s: invalid layer_id %d\n", __func__, layer_id);
        return 0;
    }

    const uint32_t strm  = seq_to_stream[seq_id];
    const auto & cells   = v_cells[strm];
    const auto & layer   = layers[it->second];
    const uint32_t il    = layer.il;

    if (!layer.v) {
        // MLA models may not have V cache
        return 0;
    }

    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    // Gather cells in logical token-position order, so imported rows map to metadata/payload position order.
    const std::vector<uint32_t> cell_idxs = llamaedge_seq_cells_by_pos(cells, seq_id);

    auto * v = layer.v_stream[strm];
    if (!v) {
        LLAMA_LOG_ERROR("%s: V tensor is null for layer %d\n", __func__, layer_id);
        return 0;
    }

    size_t expected_size = 0;
    size_t offset = 0;

    // Check if cell indices are contiguous (for fast-path batch write)
    bool contiguous = true;
    for (size_t ci = 1; ci < cell_idxs.size(); ++ci) {
        if (cell_idxs[ci] != cell_idxs[ci-1] + 1) { contiguous = false; break; }
    }

    if (!v_trans) {
        // Non-transposed V: each cell is one row
        const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
        expected_size = cell_idxs.size() * v_size_row;

        if (cell_idxs.empty() || src_size != expected_size) {
            LLAMA_LOG_ERROR("%s: V size mismatch for layer %d (need %zu, got %zu)\n",
                    __func__, layer_id, expected_size, src_size);
            return 0;
        }

        if (contiguous && !cell_idxs.empty()) {
            ggml_backend_tensor_set(v, src, cell_idxs[0] * v_size_row, cell_idxs.size() * v_size_row);
            offset = cell_idxs.size() * v_size_row;
        } else {
            for (const auto & idx : cell_idxs) {
                ggml_backend_tensor_set(v, src + offset, idx * v_size_row, v_size_row);
                offset += v_size_row;
            }
        }
    } else {
        // Transposed V: each embedding element is stored as a row
        const size_t v_size_el = ggml_type_size(v->type);
        const uint32_t kv_size = cells.size();
        expected_size = cell_idxs.size() * n_embd_v_gqa * v_size_el;

        if (cell_idxs.empty() || src_size != expected_size) {
            LLAMA_LOG_ERROR("%s: V size mismatch for layer %d (need %zu, got %zu)\n",
                    __func__, layer_id, expected_size, src_size);
            return 0;
        }

        if (contiguous && !cell_idxs.empty()) {
            // Contiguous cells: write one contiguous block per embedding dim
            const uint32_t head = cell_idxs[0];
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                const size_t dst_offset = (head + (size_t)j * kv_size) * v_size_el;
                ggml_backend_tensor_set(v, src + offset, dst_offset, cell_idxs.size() * v_size_el);
                offset += cell_idxs.size() * v_size_el;
            }
        } else {
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & idx : cell_idxs) {
                    const size_t dst_offset = (idx + (size_t)j * kv_size) * v_size_el;
                    ggml_backend_tensor_set(v, src + offset, dst_offset, v_size_el);
                    offset += v_size_el;
                }
            }
        }
    }

    return offset;
}

size_t llama_kv_cache::layer_import_k_range(
        llama_seq_id      seq_id,
        int32_t           layer_id,
        const llama_pos * pos,
        uint32_t          cell_count,
        const uint8_t   * src,
        size_t            src_size) {

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, (int) seq_id);
        return 0;
    }
    if (cell_count == 0) {
        return 0;
    }
    if (!pos || !src) {
        LLAMA_LOG_ERROR("%s: null position or source buffer\n", __func__);
        return 0;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("%s: invalid layer_id %d\n", __func__, layer_id);
        return 0;
    }

    const uint32_t strm  = seq_to_stream[seq_id];
    const auto & cells   = v_cells[strm];
    const auto & layer   = layers[it->second];
    const uint32_t il    = layer.il;

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const size_t k_size_row = ggml_row_size(layer.k->type, n_embd_k_gqa);
    const size_t expected_size = static_cast<size_t>(cell_count) * k_size_row;
    if (src_size != expected_size) {
        LLAMA_LOG_ERROR("%s: K range size mismatch for layer %d (need %zu, got %zu)\n",
                __func__, layer_id, expected_size, src_size);
        return 0;
    }

    std::vector<uint32_t> cell_idxs;
    if (!llamaedge_seq_cells_for_positions(cells, seq_id, pos, cell_count, cell_idxs)) {
        LLAMA_LOG_ERROR("%s: requested K positions not present for seq %d\n", __func__, (int) seq_id);
        return 0;
    }

    auto * k = layer.k_stream[strm];
    if (!k) {
        LLAMA_LOG_ERROR("%s: K tensor is null for layer %d\n", __func__, layer_id);
        return 0;
    }

    bool contiguous = true;
    for (size_t ci = 1; ci < cell_idxs.size(); ++ci) {
        if (cell_idxs[ci] != cell_idxs[ci - 1] + 1) {
            contiguous = false;
            break;
        }
    }

    size_t offset = 0;
    if (contiguous) {
        ggml_backend_tensor_set(k, src, static_cast<size_t>(cell_idxs[0]) * k_size_row, expected_size);
        offset = expected_size;
    } else {
        for (const auto & idx : cell_idxs) {
            ggml_backend_tensor_set(k, src + offset, static_cast<size_t>(idx) * k_size_row, k_size_row);
            offset += k_size_row;
        }
    }

    return offset;
}

size_t llama_kv_cache::layer_import_v_range(
        llama_seq_id      seq_id,
        int32_t           layer_id,
        const llama_pos * pos,
        uint32_t          cell_count,
        const uint8_t   * src,
        size_t            src_size) {

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        LLAMA_LOG_ERROR("%s: invalid seq_id %d\n", __func__, (int) seq_id);
        return 0;
    }
    if (cell_count == 0) {
        return 0;
    }
    if (!pos || !src) {
        LLAMA_LOG_ERROR("%s: null position or source buffer\n", __func__);
        return 0;
    }

    auto it = map_layer_ids.find(layer_id);
    if (it == map_layer_ids.end()) {
        LLAMA_LOG_ERROR("%s: invalid layer_id %d\n", __func__, layer_id);
        return 0;
    }

    const uint32_t strm  = seq_to_stream[seq_id];
    const auto & cells   = v_cells[strm];
    const auto & layer   = layers[it->second];
    const uint32_t il    = layer.il;

    if (!layer.v) {
        return 0;
    }

    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);
    std::vector<uint32_t> cell_idxs;
    if (!llamaedge_seq_cells_for_positions(cells, seq_id, pos, cell_count, cell_idxs)) {
        LLAMA_LOG_ERROR("%s: requested V positions not present for seq %d\n", __func__, (int) seq_id);
        return 0;
    }

    auto * v = layer.v_stream[strm];
    if (!v) {
        LLAMA_LOG_ERROR("%s: V tensor is null for layer %d\n", __func__, layer_id);
        return 0;
    }

    bool contiguous = true;
    for (size_t ci = 1; ci < cell_idxs.size(); ++ci) {
        if (cell_idxs[ci] != cell_idxs[ci - 1] + 1) {
            contiguous = false;
            break;
        }
    }

    size_t expected_size = 0;
    size_t offset = 0;
    if (!v_trans) {
        const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
        expected_size = static_cast<size_t>(cell_count) * v_size_row;
        if (src_size != expected_size) {
            LLAMA_LOG_ERROR("%s: V range size mismatch for layer %d (need %zu, got %zu)\n",
                    __func__, layer_id, expected_size, src_size);
            return 0;
        }
        if (contiguous) {
            ggml_backend_tensor_set(v, src, static_cast<size_t>(cell_idxs[0]) * v_size_row, expected_size);
            offset = expected_size;
        } else {
            for (const auto & idx : cell_idxs) {
                ggml_backend_tensor_set(v, src + offset, static_cast<size_t>(idx) * v_size_row, v_size_row);
                offset += v_size_row;
            }
        }
    } else {
        // Transposed V expects src in [embedding_dim][token_position] order.
        const size_t v_size_el = ggml_type_size(v->type);
        const uint32_t kv_size = cells.size();
        expected_size = static_cast<size_t>(cell_count) * n_embd_v_gqa * v_size_el;
        if (src_size != expected_size) {
            LLAMA_LOG_ERROR("%s: V range size mismatch for layer %d (need %zu, got %zu)\n",
                    __func__, layer_id, expected_size, src_size);
            return 0;
        }
        if (contiguous) {
            const uint32_t head = cell_idxs[0];
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                const size_t dst_offset = (head + static_cast<size_t>(j) * kv_size) * v_size_el;
                ggml_backend_tensor_set(v, src + offset, dst_offset, static_cast<size_t>(cell_count) * v_size_el);
                offset += static_cast<size_t>(cell_count) * v_size_el;
            }
        } else {
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & idx : cell_idxs) {
                    const size_t dst_offset = (idx + static_cast<size_t>(j) * kv_size) * v_size_el;
                    ggml_backend_tensor_set(v, src + offset, dst_offset, v_size_el);
                    offset += v_size_el;
                }
            }
        }
    }

    return offset;
}

#endif // LLAMAEDGE_ENABLE_KV_LAYER_IMPORT
