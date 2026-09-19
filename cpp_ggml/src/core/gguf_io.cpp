#include "core/gguf_io.hpp"

#include <cstdio>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace instantmesh {

GgufModel::~GgufModel() {
    unload();
}

void GgufModel::unload() {
    if (buffer) ggml_backend_buffer_free(buffer);
    buffer = nullptr;
    if (gguf) gguf_free(gguf);
    gguf = nullptr;
    if (ctx) ggml_free(ctx);
    ctx = nullptr;
    tensors.clear();
}

uint32_t kv_u32(const gguf_context * g, const char * key, uint32_t def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (uint32_t) gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return (uint32_t) gguf_get_val_i64(g, id);
        case GGUF_TYPE_FLOAT32:return (uint32_t) gguf_get_val_f32(g, id);
        default: return def;
    }
}
int32_t kv_i32(const gguf_context * g, const char * key, int32_t def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (int32_t) gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int32_t) gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return (int32_t) gguf_get_val_i64(g, id);
        case GGUF_TYPE_FLOAT32:return (int32_t) gguf_get_val_f32(g, id);
        default: return def;
    }
}
float kv_f32(const gguf_context * g, const char * key, float def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(g, id);
        case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(g, id);
        case GGUF_TYPE_INT32:   return (float) gguf_get_val_i32(g, id);
        case GGUF_TYPE_INT64:   return (float) gguf_get_val_i64(g, id);
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(g, id) ? 1.0f : 0.0f;
        default: return def;
    }
}
bool kv_bool(const gguf_context * g, const char * key, bool def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(g, id);
        case GGUF_TYPE_INT32:   return gguf_get_val_i32(g, id) != 0;
        case GGUF_TYPE_INT64:   return gguf_get_val_i64(g, id) != 0;
        default: return def;
    }
}
const char * kv_str(const gguf_context * g, const char * key, const char * def) {
    const int64_t id = gguf_find_key(g, key);
    return id < 0 ? def : gguf_get_val_str(g, id);
}

bool load_gguf(const std::string & path, ggml_backend_t backend,
               GgufModel & out, std::string * error) {
    // Parse metadata only; weights are streamed from file after allocation.
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = &out.ctx;

    out.gguf = gguf_init_from_file(path.c_str(), params);
    if (!out.gguf) {
        if (error) *error = std::string("gguf_init_from_file failed: ") + path;
        return false;
    }

    // Index tensors by name for O(1) graph wiring later.
    const int n_tensors = gguf_get_n_tensors(out.gguf);
    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = ggml_get_tensor(out.ctx, gguf_get_tensor_name(out.gguf, i));
        if (t) out.tensors.emplace(ggml_get_name(t), t);
    }

    // Allocate every tensor on the chosen backend (single buffer).
    out.buffer = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (out.buffer == nullptr) {
        if (error) *error = "ggml_backend_alloc_ctx_tensors failed: " + path;
        return false;
    }

    // Stream weight payloads from the file into the backend memory.
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (error) *error = "fopen failed: " + path;
        return false;
    }
    std::vector<uint8_t> buf;
    for (int i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = out.tensors[gguf_get_tensor_name(out.gguf, i)];
        const size_t offs = gguf_get_data_offset(out.gguf) + gguf_get_tensor_offset(out.gguf, i);
        const size_t nbytes = ggml_nbytes(t);
        buf.resize(nbytes);
        if (std::fseek(f, static_cast<long>(offs), SEEK_SET) != 0 ||
            std::fread(buf.data(), 1, nbytes, f) != nbytes) {
            std::fclose(f);
            if (error) *error = "read failed for tensor: " + std::string(ggml_get_name(t));
            return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    }
    std::fclose(f);
    return true;
}

} // namespace instantmesh