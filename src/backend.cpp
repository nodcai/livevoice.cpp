#include "backend.hpp"

#include "common.hpp"

#include "ggml-cpu.h"

#include <cctype>
#include <cstdlib>
#include <string>

namespace lv {

namespace {

ggml_backend_t g_backend = nullptr;
ggml_gallocr_t g_gallocr = nullptr;
std::string    g_name;
std::once_flag g_once;

std::string lower(const char* s) {
    std::string r = s ? s : "";
    for (auto& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

// A registered device whose name contains `want`, or the first device of the
// matching type for the generic names.
ggml_backend_t init_named(const std::string& want) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev && lower(ggml_backend_dev_name(dev)).find(want) != std::string::npos)
            if (ggml_backend_t b = ggml_backend_dev_init(dev, nullptr)) return b;
    }
    if (want == "cpu" || want == "gpu") {
        const auto type = want == "cpu" ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU;
        if (ggml_backend_dev_t dev = ggml_backend_dev_by_type(type)) return ggml_backend_dev_init(dev, nullptr);
    }
    return nullptr;
}

void init() {
    ggml_backend_load_all();

    if (const char* env = std::getenv("LIVEVOICE_BACKEND")) {
        g_backend = init_named(lower(env));
        if (!g_backend) LV_LOG_WARN("backend: LIVEVOICE_BACKEND=%s has no matching device, using the default", env);
    }
    if (!g_backend) g_backend = ggml_backend_init_best();
    if (!g_backend) g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!g_backend) {
        LV_LOG_ERROR("backend: no ggml backend could be initialised");
        g_name = "none";
        return;
    }

    g_name = ggml_backend_name(g_backend);
    if (ggml_backend_is_cpu(g_backend)) ggml_backend_cpu_set_n_threads(g_backend, 0);
    g_gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    LV_LOG_INFO("backend: %s", g_name.c_str());
    // Free the allocator and backend before the backend library's own static
    // destructors run: ggml-metal asserts that no residency set is alive when
    // its device is torn down, and this handler (registered after that
    // library initialised) runs first.
    std::atexit([] {
        if (g_gallocr) { ggml_gallocr_free(g_gallocr); g_gallocr = nullptr; }
        if (g_backend) { ggml_backend_free(g_backend); g_backend = nullptr; }
    });
}

}  // namespace

ggml_backend_t backend() {
    std::call_once(g_once, init);
    return g_backend;
}

const char* backend_name() {
    std::call_once(g_once, init);
    return g_name.c_str();
}

bool compute_graph(ggml_cgraph* graph) {
    ggml_backend_t b = backend();
    if (!b || !graph || !g_gallocr) return false;
    if (!ggml_gallocr_alloc_graph(g_gallocr, graph)) {
        LV_LOG_ERROR("backend: graph allocation failed");
        return false;
    }
    return ggml_backend_graph_compute(b, graph) == GGML_STATUS_SUCCESS;
}

ggml_backend_buffer_t allocate_ctx_tensors(ggml_context* ctx) {
    ggml_backend_t b = backend();
    if (!b || !ctx) return nullptr;
    return ggml_backend_alloc_ctx_tensors(ctx, b);
}

bool backend_supports_flash_attn() {
    static std::once_flag once;
    static bool supported = false;
    std::call_once(once, [] {
        const std::string env = lower(std::getenv("LIVEVOICE_FLASH_ATTN"));
        if (env == "0" || env == "off" || env == "false") {
            LV_LOG_INFO("backend: flash attention off (LIVEVOICE_FLASH_ATTN=%s)", env.c_str());
            return;
        }
        ggml_backend_t b = backend();
        if (!b) return;
        // Ask the backend about a representative op; nothing is allocated.
        struct ggml_init_params ip {};
        ip.mem_size = ggml_tensor_overhead() * 8;
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 14, 1);
        ggml_tensor* k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 16, 2, 1);
        ggml_tensor* v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 16, 2, 1);
        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 16, 1);
        ggml_tensor* op   = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
        supported = op && ggml_backend_supports_op(b, op);
        ggml_free(ctx);
        LV_LOG_INFO("backend: flash attention %s (%s)", supported ? "on" : "unavailable", g_name.c_str());
    });
    return supported;
}

std::mutex& compute_mutex() {
    static std::mutex mu;
    return mu;
}

}  // namespace lv
