#ifndef LIVEVOICE_BACKEND_HPP
#define LIVEVOICE_BACKEND_HPP

// Process-wide ggml backend and graph compute.
//
// The first call picks one backend for the process lifetime:
//   1. LIVEVOICE_BACKEND (case-insensitive: metal, cuda, vulkan, cpu, ...)
//      when a matching device is registered,
//   2. otherwise the best available device, GPU before CPU.
//
// Graph intermediates come from one shared allocator, so graphs must not be
// computed from two threads at once; generation holds compute_mutex().

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <mutex>

namespace lv {

ggml_backend_t backend();
const char*    backend_name();

// Allocates the graph's intermediates and runs it on the backend. All leaf
// tensors (weights, inputs, caches) must already live in backend buffers.
bool compute_graph(ggml_cgraph* graph);

// Allocates every tensor of a no_alloc context in one backend buffer. The
// caller owns the buffer.
ggml_backend_buffer_t allocate_ctx_tensors(ggml_context* ctx);

// True when the backend runs ggml_flash_attn_ext natively. LIVEVOICE_FLASH_ATTN=0
// turns it off.
bool backend_supports_flash_attn();

std::mutex& compute_mutex();

}  // namespace lv

#endif  // LIVEVOICE_BACKEND_HPP
