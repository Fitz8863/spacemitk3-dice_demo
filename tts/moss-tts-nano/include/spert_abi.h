// SPDX-FileCopyrightText: Copyright (c) 2026 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

// include/spert_abi.h - compiler and MLIR integration ABI.

#pragma once
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#define MLIR_RUNNERUTILS_EXPORT __attribute__((visibility("default")))
#define SPINE_CACHE_LINE        64
#define SPINE_CACHE_ALIGN       __attribute__((aligned(SPINE_CACHE_LINE)))

extern "C" {
/// Returns the architecture identifier of the default backend's CC cores.
/// Returns zero when the default backend is unavailable.
int64_t spine_get_current_arch_id() noexcept;

/// Returns the per-core shared-memory capacity of the default backend in bytes.
/// Returns zero when the default backend is unavailable.
int64_t spine_get_ai_core_tcm_size() noexcept;

/// Returns the vector length reported for the default backend's CC cores.
/// Returns zero when the default backend is unavailable.
int64_t spine_get_ai_core_vlen() noexcept;

/// Returns the number of CC cores exposed by the default backend.
/// Returns zero when the default backend is unavailable.
int64_t spine_get_num_cores() noexcept;

/// Acquires a stream using all available CC cores of the default backend.
///
/// The returned nonzero handle is owned by the caller and must be released with
/// `spine_release_stream`. Returns zero when stream creation fails.
int64_t spine_require_stream() noexcept;

/// Acquires a stream with a requested CC-core configuration.
///
/// `num_cores` must fit in `uint32_t`; zero requests all available cores. When
/// `core_mask` is non-null and `num_cores` is positive, its entries are treated
/// as preferred physical core identifiers. If that exact set cannot be granted,
/// the runtime falls back to the requested core count. Returns zero on invalid
/// input or stream-creation failure. The caller owns every nonzero handle.
int64_t spine_require_stream_with_config(int64_t num_cores, int64_t * core_mask) noexcept;

/// Releases a stream handle previously returned by `spine_require_stream` or
/// `spine_require_stream_with_config`. A zero handle is ignored.
void spine_release_stream(int64_t stream) noexcept;

// Dispatch functions invoke `function` as
// `void(int64_t context, void *function_args)` once per grid element.
// `function_args` is not copied: it must remain valid until a synchronous call
// returns or until the asynchronous token is consumed by `spine_parallel_sync`.
// Every grid dimension must be positive and fit in `uint32_t`.
//
/// Launches a one-dimensional grid and waits for all tiles to terminate.
///
/// `function` must point to a function compatible with
/// `void(int64_t context, void *function_args)`. Invalid input or launch failure
/// is ignored because this compatibility ABI has no status return value.
void spine_parallel_dispatch_1d(int64_t stream,         //
                                void *  function,       //
                                void *  function_args,  //
                                int64_t grid_x_size) noexcept;

/// Launches a two-dimensional grid and waits for all tiles to terminate.
void spine_parallel_dispatch_2d(int64_t stream,         //
                                void *  function,       //
                                void *  function_args,  //
                                int64_t grid_x_size,    //
                                int64_t grid_y_size) noexcept;

/// Launches a three-dimensional grid and waits for all tiles to terminate.
void spine_parallel_dispatch_3d(int64_t stream,         //
                                void *  function,       //
                                void *  function_args,  //
                                int64_t grid_x_size,    //
                                int64_t grid_y_size,    //
                                int64_t grid_z_size) noexcept;

/// Launches a one-dimensional grid without waiting.
///
/// Returns an owned asynchronous token, or zero on invalid input, allocation
/// failure, or launch failure. A nonzero token must be consumed exactly once by
/// `spine_parallel_sync` using the stream that created it.
int64_t spine_parallel_dispatch_1d_async(int64_t stream,         //
                                         void *  function,       //
                                         void *  function_args,  //
                                         int64_t grid_x_size) noexcept;

/// Launches a two-dimensional grid without waiting.
/// Returns an owned asynchronous token, or zero on failure.
int64_t spine_parallel_dispatch_2d_async(int64_t stream,         //
                                         void *  function,       //
                                         void *  function_args,  //
                                         int64_t grid_x_size,    //
                                         int64_t grid_y_size) noexcept;

/// Launches a three-dimensional grid without waiting.
/// Returns an owned asynchronous token, or zero on failure.
int64_t spine_parallel_dispatch_3d_async(int64_t stream,         //
                                         void *  function,       //
                                         void *  function_args,  //
                                         int64_t grid_x_size,    //
                                         int64_t grid_y_size,    //
                                         int64_t grid_z_size) noexcept;

/// Waits for and consumes asynchronous tokens created by `stream`.
///
/// For each of the `tile_size` entries in `tile`, a valid token belonging to
/// `stream` is awaited, released, and replaced with zero. Null tokens and tokens
/// from another stream are ignored. Tokens must be unique and unconsumed;
/// passing duplicate, stale, or otherwise invalid nonzero values is unsupported.
void spine_parallel_sync(int64_t stream, int64_t * tile, int64_t tile_size) noexcept;

/// Returns the calling tile's zero-based coordinate on `axis`.
///
/// `ctx` must be the context handle supplied to the running kernel and `axis`
/// must be in `[0, 2]`. Returns zero for invalid input.
int64_t spine_grid(int64_t ctx, int64_t axis) noexcept;

/// Allocates `size` bytes from the calling tile's worker-local shared memory.
///
/// The call may suspend the tile until space is available. Returns null for an
/// invalid context, a zero-size request, or allocation failure. The allocation
/// must be released by the same running tile with `spine_thread_tcm_free`.
void * spine_thread_tcm_malloc(int64_t ctx, size_t size) noexcept;

/// Releases a pointer returned by `spine_thread_tcm_malloc` for the same tile.
/// Null context or pointer values are ignored.
void spine_thread_tcm_free(int64_t ctx, void * ptr) noexcept;

/// Weak allocation hook for integrations that associate memory with an ID.
///
/// The default implementation ignores `ctx`, `id`, and `type` and delegates to
/// `malloc(size)`. A strong definition may override this symbol.
void * spine_alloc_with_id(int64_t ctx, int64_t id, size_t size, int64_t type) noexcept;

/// Weak deallocation hook paired with `spine_alloc_with_id`.
/// The default implementation delegates to `free(ptr)`.
void spine_free_with_id(int64_t ctx, int64_t id, void * ptr, int64_t type) noexcept;
}

/// Issues an architecture-appropriate CPU relaxation hint.
///
/// The operation may yield execution on unsupported architectures and does not
/// provide synchronization by itself.
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
static inline void spine_thread_cpu_relax(void) noexcept {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__riscv)
static inline void spine_thread_cpu_relax(void) noexcept {
    __asm__ volatile("pause " ::: "memory");
}
#else
static inline void spine_thread_cpu_relax(void) noexcept {
    std::this_thread::yield();
}
#endif
