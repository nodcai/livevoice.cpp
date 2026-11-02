#ifndef LIVEVOICE_GGUF_LOADER_HPP
#define LIVEVOICE_GGUF_LOADER_HPP

// gguf reader: name -> tensor map plus typed metadata accessors.
//
// load() puts the tensor data in a buffer of the active backend (device
// memory on a GPU); load_host() keeps it in host memory, for files that are
// only read once and copied elsewhere (voice prompts).

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace lv {

class GgufLoader {
public:
    GgufLoader() = default;
    ~GgufLoader();
    GgufLoader(const GgufLoader&)            = delete;
    GgufLoader& operator=(const GgufLoader&) = delete;

    bool load(const std::string& path);
    bool load_host(const std::string& path);

    struct ggml_tensor* tensor(const std::string& name) const;
    bool                has(const std::string& name) const { return tensor(name) != nullptr; }
    const std::vector<std::string>& tensor_names() const { return names_; }

    // ggml's element-wise ops do not mix F32 and F16 operands, so norm
    // scales, biases and other small F16 tensors are promoted to F32 once at
    // load time. Matmul weights stay as they are.
    void promote_small_f16_to_f32(size_t max_elems = 65536);

    // Metadata; the default is returned when the key is absent or of
    // another type.
    int32_t     get_i32 (const std::string& key, int32_t def = 0) const;
    uint32_t    get_u32 (const std::string& key, uint32_t def = 0) const;
    float       get_f32 (const std::string& key, float def = 0.0f) const;
    bool        get_bool(const std::string& key, bool def = false) const;
    std::string get_str (const std::string& key, const std::string& def = {}) const;
    std::vector<int32_t>     get_i32_array(const std::string& key) const;
    std::vector<std::string> get_str_array(const std::string& key) const;

private:
    bool open(const std::string& path, bool host);

    struct gguf_context*  gguf_        = nullptr;
    struct ggml_context*  ctx_         = nullptr;
    struct ggml_context*  promote_ctx_ = nullptr;
    ggml_backend_buffer_t buf_         = nullptr;
    ggml_backend_buffer_t promote_buf_ = nullptr;
    std::vector<std::string> names_;
    std::unordered_map<std::string, struct ggml_tensor*> by_name_;
};

// Any F32 / F16 / quantized tensor, from a backend buffer or host memory,
// as host F32.
std::vector<float> tensor_to_f32(const struct ggml_tensor* t);

}  // namespace lv

#endif  // LIVEVOICE_GGUF_LOADER_HPP
