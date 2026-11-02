#include "gguf_loader.hpp"

#include "backend.hpp"
#include "common.hpp"

#include <cstdio>
#include <cstring>

namespace lv {

GgufLoader::~GgufLoader() {
    if (promote_buf_) ggml_backend_buffer_free(promote_buf_);
    if (promote_ctx_) ggml_free(promote_ctx_);
    if (buf_)         ggml_backend_buffer_free(buf_);
    if (gguf_)        gguf_free(gguf_);
    if (ctx_)         ggml_free(ctx_);
}

bool GgufLoader::load(const std::string& path)      { return open(path, Mode::Backend); }
bool GgufLoader::load_host(const std::string& path) { return open(path, Mode::Host); }
bool GgufLoader::load_meta(const std::string& path) { return open(path, Mode::Meta); }

bool GgufLoader::open(const std::string& path, Mode mode) {
    if (gguf_) {
        LV_LOG_ERROR("gguf: loader already holds a file");
        return false;
    }
    struct gguf_init_params p {};
    p.no_alloc = mode != Mode::Host;
    p.ctx      = mode == Mode::Meta ? nullptr : &ctx_;
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_) {
        LV_LOG_ERROR("gguf: cannot read %s", path.c_str());
        return false;
    }
    if (mode == Mode::Meta) return true;
    const int64_t n = gguf_get_n_tensors(gguf_);
    names_.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        if (struct ggml_tensor* t = name ? ggml_get_tensor(ctx_, name) : nullptr) {
            names_.emplace_back(name);
            by_name_.emplace(name, t);
        }
    }
    if (mode == Mode::Host || n == 0) return true;

    // Stream the tensor bytes from the file into one backend buffer.
    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend());
    if (!buf_) {
        LV_LOG_ERROR("gguf: backend allocation failed for %s", path.c_str());
        return false;
    }
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        LV_LOG_ERROR("gguf: cannot open %s", path.c_str());
        return false;
    }
    const size_t data_off = gguf_get_data_offset(gguf_);
    std::vector<uint8_t> stage;
    bool ok = true;
    for (int64_t i = 0; i < n && ok; ++i) {
        struct ggml_tensor* t = tensor(gguf_get_tensor_name(gguf_, i));
        const size_t sz = gguf_get_tensor_size(gguf_, i);
        if (!t || sz == 0) continue;
        if (stage.size() < sz) stage.resize(sz);
        ok = std::fseek(fp, static_cast<long>(data_off + gguf_get_tensor_offset(gguf_, i)), SEEK_SET) == 0 &&
             std::fread(stage.data(), 1, sz, fp) == sz;
        if (ok) ggml_backend_tensor_set(t, stage.data(), 0, sz);
    }
    std::fclose(fp);
    if (!ok) {
        LV_LOG_ERROR("gguf: short read in %s", path.c_str());
        return false;
    }
    LV_LOG_INFO("gguf: %s, %lld tensors on %s", path.c_str(), static_cast<long long>(n), backend_name());
    return true;
}

struct ggml_tensor* GgufLoader::tensor(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : it->second;
}

void GgufLoader::promote_small_f16_to_f32(size_t max_elems) {
    std::vector<std::string> keys;
    for (const auto& kv : by_name_)
        if (kv.second->type == GGML_TYPE_F16 && static_cast<size_t>(ggml_nelements(kv.second)) <= max_elems)
            keys.push_back(kv.first);
    if (keys.empty() || promote_ctx_) return;

    struct ggml_init_params ip {};
    ip.mem_size = (keys.size() + 1) * ggml_tensor_overhead();
    ip.no_alloc = true;
    promote_ctx_ = ggml_init(ip);
    // The gguf-owned context is sized exactly to the file, so promoted
    // tensors need a sibling context of their own.
    std::vector<struct ggml_tensor*> fresh;
    for (const auto& k : keys) {
        const struct ggml_tensor* t = by_name_[k];
        struct ggml_tensor* nt = ggml_new_tensor(promote_ctx_, GGML_TYPE_F32, ggml_n_dims(t), t->ne);
        ggml_set_name(nt, k.c_str());
        fresh.push_back(nt);
    }
    promote_buf_ = ggml_backend_alloc_ctx_tensors(promote_ctx_, backend());
    if (!promote_buf_) {
        LV_LOG_ERROR("gguf: promotion buffer allocation failed");
        return;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::vector<float> v = tensor_to_f32(by_name_[keys[i]]);
        ggml_backend_tensor_set(fresh[i], v.data(), 0, v.size() * sizeof(float));
        by_name_[keys[i]] = fresh[i];
    }
}

namespace {
int64_t find(struct gguf_context* g, const std::string& key) { return g ? gguf_find_key(g, key.c_str()) : -1; }
}  // namespace

int32_t GgufLoader::get_i32(const std::string& key, int32_t def) const {
    const int64_t k = find(gguf_, key);
    if (k < 0) return def;
    switch (gguf_get_kv_type(gguf_, k)) {
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(gguf_, k);
        case GGUF_TYPE_UINT32: return static_cast<int32_t>(gguf_get_val_u32(gguf_, k));
        default:               return def;
    }
}

uint32_t GgufLoader::get_u32(const std::string& key, uint32_t def) const {
    return static_cast<uint32_t>(get_i32(key, static_cast<int32_t>(def)));
}

float GgufLoader::get_f32(const std::string& key, float def) const {
    const int64_t k = find(gguf_, key);
    if (k < 0) return def;
    switch (gguf_get_kv_type(gguf_, k)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf_, k);
        case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(gguf_, k));
        default:                return def;
    }
}

bool GgufLoader::get_bool(const std::string& key, bool def) const {
    const int64_t k = find(gguf_, key);
    return (k >= 0 && gguf_get_kv_type(gguf_, k) == GGUF_TYPE_BOOL) ? gguf_get_val_bool(gguf_, k) : def;
}

std::string GgufLoader::get_str(const std::string& key, const std::string& def) const {
    const int64_t k = find(gguf_, key);
    if (k < 0 || gguf_get_kv_type(gguf_, k) != GGUF_TYPE_STRING) return def;
    const char* s = gguf_get_val_str(gguf_, k);
    return s ? s : def;
}

std::vector<int32_t> GgufLoader::get_i32_array(const std::string& key) const {
    std::vector<int32_t> out;
    const int64_t k = find(gguf_, key);
    if (k < 0 || gguf_get_kv_type(gguf_, k) != GGUF_TYPE_ARRAY) return out;
    const enum gguf_type at = gguf_get_arr_type(gguf_, k);
    if (at != GGUF_TYPE_INT32 && at != GGUF_TYPE_UINT32) return out;
    out.resize(gguf_get_arr_n(gguf_, k));
    if (!out.empty()) std::memcpy(out.data(), gguf_get_arr_data(gguf_, k), out.size() * sizeof(int32_t));
    return out;
}

std::vector<std::string> GgufLoader::get_str_array(const std::string& key) const {
    std::vector<std::string> out;
    const int64_t k = find(gguf_, key);
    if (k < 0 || gguf_get_kv_type(gguf_, k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf_, k) != GGUF_TYPE_STRING)
        return out;
    const size_t n = gguf_get_arr_n(gguf_, k);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const char* s = gguf_get_arr_str(gguf_, k, i);
        out.emplace_back(s ? s : "");
    }
    return out;
}

std::vector<float> tensor_to_f32(const struct ggml_tensor* t) {
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    std::vector<uint8_t> raw(ggml_nbytes(t));
    if (t->buffer) ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    else           std::memcpy(raw.data(), t->data, raw.size());
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), raw.data(), n * sizeof(float));
    } else if (const auto* tt = ggml_get_type_traits(t->type); tt && tt->to_float) {
        tt->to_float(raw.data(), out.data(), static_cast<int64_t>(n));
    } else {
        LV_LOG_ERROR("gguf: cannot convert %s (%s) to f32", t->name, ggml_type_name(t->type));
        out.clear();
    }
    return out;
}

}  // namespace lv
