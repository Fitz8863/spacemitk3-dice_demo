// SPDX-FileCopyrightText: Copyright (c) 2026 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

// include/spert_engine.hpp - internal type-erased engine ABI.
//
// This is the boundary between the header-only templated public API
// (spert.hpp, compiled in the user's translation unit) and the precompiled
// runtime engine. The public templates pack a user
// kernel + its arguments into a heap closure and hand the engine a
// type-erased (invoker, closure, deleter) triple. No template code crosses
// into the engine, so the ABI stays stable regardless of kernel signature.
//
// This header is installed because spert.hpp depends on its declarations.
// Names in spert::detail are implementation ABI, not source-level public API.

#pragma once

#include <cstddef>
#include <cstdint>

namespace spert {
class Context;  // opaque to users; aliases detail::Tile*

/// Terminal and diagnostic outcomes returned by runtime operations.
///
/// Numeric values are part of the engine ABI and must remain stable within the
/// current shared-library ABI version.
enum class Status : int {
    Ok                 = 0,  ///< Operation completed successfully.
    InvalidArg         = 1,  ///< A handle, pointer, dimension, or value was invalid.
    Deadlock           = 2,  ///< The stream reached a confirmed cooperative deadlock.
    Timeout            = 3,  ///< A caller-supplied deadline expired before completion.
    BackendUnavailable = 4,  ///< The selected backend is unavailable or shutting down.
    Unsupported        = 5,  ///< The requested operation is not supported.
    NoMem              = 6,  ///< Runtime storage could not be allocated.
    Failed             = 7,  ///< The operation failed without a more specific status.
    Cancelled          = 8,  ///< Work was abandoned during stream teardown.
    CrossStream        = 9,  ///< A stream-scoped object was used from another stream.
};

/// Converts a status value to its stable ABI integer representation.
constexpr int status_code(Status status) noexcept {
    return static_cast<int>(status);
}

constexpr bool is_valid_status_code(int code) noexcept {
    return code >= status_code(Status::Ok) && code <= status_code(Status::CrossStream);
}

constexpr Status decode_status_code(int code) noexcept {
    return is_valid_status_code(code) ? static_cast<Status>(code) : Status::Failed;
}

inline constexpr int kFuturePending = 0;

/// Encodes a terminal status for storage in a future completion word.
/// Zero remains reserved for the pending state.
constexpr int encode_terminal(Status status) noexcept {
    return status_code(status) + 1;
}

constexpr bool is_valid_terminal(int terminal) noexcept {
    return terminal >= encode_terminal(Status::Ok) && terminal <= encode_terminal(Status::CrossStream);
}

constexpr Status decode_terminal(int terminal) noexcept {
    return is_valid_terminal(terminal) ? decode_status_code(terminal - 1) : Status::Failed;
}

static_assert(status_code(Status::Ok) == 0);
static_assert(status_code(Status::InvalidArg) == 1);
static_assert(status_code(Status::Deadlock) == 2);
static_assert(status_code(Status::Timeout) == 3);
static_assert(status_code(Status::BackendUnavailable) == 4);
static_assert(status_code(Status::Unsupported) == 5);
static_assert(status_code(Status::NoMem) == 6);
static_assert(status_code(Status::Failed) == 7);
static_assert(status_code(Status::Cancelled) == 8);
static_assert(status_code(Status::CrossStream) == 9);

static_assert(kFuturePending == 0);
static_assert(encode_terminal(Status::Ok) == 1);
static_assert(encode_terminal(Status::InvalidArg) == 2);
static_assert(encode_terminal(Status::Deadlock) == 3);
static_assert(encode_terminal(Status::Timeout) == 4);
static_assert(encode_terminal(Status::BackendUnavailable) == 5);
static_assert(encode_terminal(Status::Unsupported) == 6);
static_assert(encode_terminal(Status::NoMem) == 7);
static_assert(encode_terminal(Status::Failed) == 8);
static_assert(encode_terminal(Status::Cancelled) == 9);
static_assert(encode_terminal(Status::CrossStream) == 10);

static_assert(!is_valid_status_code(-1));
static_assert(is_valid_status_code(status_code(Status::Ok)));
static_assert(is_valid_status_code(status_code(Status::CrossStream)));
static_assert(!is_valid_status_code(status_code(Status::CrossStream) + 1));
static_assert(!is_valid_terminal(kFuturePending));
static_assert(is_valid_terminal(encode_terminal(Status::Ok)));
static_assert(is_valid_terminal(encode_terminal(Status::CrossStream)));
static_assert(!is_valid_terminal(encode_terminal(Status::CrossStream) + 1));

static_assert(decode_status_code(0) == Status::Ok);
static_assert(decode_status_code(1) == Status::InvalidArg);
static_assert(decode_status_code(2) == Status::Deadlock);
static_assert(decode_status_code(3) == Status::Timeout);
static_assert(decode_status_code(4) == Status::BackendUnavailable);
static_assert(decode_status_code(5) == Status::Unsupported);
static_assert(decode_status_code(6) == Status::NoMem);
static_assert(decode_status_code(7) == Status::Failed);
static_assert(decode_status_code(8) == Status::Cancelled);
static_assert(decode_status_code(9) == Status::CrossStream);

static_assert(decode_terminal(1) == Status::Ok);
static_assert(decode_terminal(2) == Status::InvalidArg);
static_assert(decode_terminal(3) == Status::Deadlock);
static_assert(decode_terminal(4) == Status::Timeout);
static_assert(decode_terminal(5) == Status::BackendUnavailable);
static_assert(decode_terminal(6) == Status::Unsupported);
static_assert(decode_terminal(7) == Status::NoMem);
static_assert(decode_terminal(8) == Status::Failed);
static_assert(decode_terminal(9) == Status::Cancelled);
static_assert(decode_terminal(10) == Status::CrossStream);
static_assert(decode_status_code(-1) == Status::Failed);
static_assert(decode_status_code(10) == Status::Failed);
static_assert(decode_terminal(kFuturePending) == Status::Failed);
static_assert(decode_terminal(11) == Status::Failed);

/// Point-in-time accounting for resources retained by the default runtime.
struct RuntimeStats {
    size_t   tiles_allocated    = 0;
    size_t   tiles_free         = 0;
    size_t   futures_allocated  = 0;
    size_t   futures_free       = 0;
    size_t   barriers_allocated = 0;
    size_t   barriers_free      = 0;
    size_t   events_allocated   = 0;
    size_t   active_streams     = 0;
    size_t   retired_streams    = 0;
    uint64_t live_tiles         = 0;
    uint64_t waiting_tiles      = 0;
};

/// A non-owning view of worker-local shared scratch memory.
///
/// The backing storage may be platform TCM or a runtime fallback allocation.
/// The view does not expose or transfer ownership of that storage.
struct SharedBufferView {
    void * data = nullptr;
    size_t size = 0;

    explicit operator bool() const { return data != nullptr && size != 0; }
};

}  // namespace spert

namespace spert::detail {

// Opaque engine types defined by the precompiled runtime.
struct Future;
struct Barrier;
struct Event;
struct Tile;
struct BackendInstance;

/// Invokes one tile using a runtime context and launch-owned closure.
/// `ctx` is a `detail::Tile*` represented to public code as `spert::Context*`.
using TileInvoker                           = void (*)(void * ctx, void * closure);
/// Destroys a launch closure exactly once after terminal completion.
using ClosureDeleter                        = void (*)(void * closure);
/// Moves a closure into runtime-owned inline storage without throwing.
using ClosureMover                          = void (*)(void * dst, void * src) noexcept;
inline constexpr size_t kInlineClosureBytes = 64;

// ---- runtime init ----
struct BackendInfo {
    size_t  shared_mem_size = 0;
    size_t  vlen            = 0;
    size_t  num_cores       = 0;
    int64_t core_arch_id    = 0;
};

/// Returns a process-lifetime backend instance for `name`, or `nullptr` when
/// the name is unknown or initialization fails.
BackendInstance * backend_get(const char * name);
/// Returns the lazily selected default backend instance, or `nullptr` on
/// selection or initialization failure.
BackendInstance * default_backend_instance();
/// Returns a zero-initialized record when the default backend is unavailable.
BackendInfo       backend_info();
/// Returns a zero-initialized record when `instance` is invalid or incomplete.
BackendInfo       backend_info(BackendInstance * instance);
/// Samples accounting for resources retained by the default runtime.
RuntimeStats      runtime_stats();

// There is no public Session in the ABI. Each backend instance owns one
// WorkerPool and arena, constructed lazily on first use and torn down at process
// exit. The no-argument helpers below continue to target the default backend.
uint32_t pool_worker_count();

// ---- stream lifecycle (per-core checkout; execution-affinity scope) ----
// A stream checks out a subset of CC cores and routes its launched tiles only
// to workers bound to those cores. `n_cores == 0` requests all available cores.
struct StreamState;  // opaque
/// Creates a stream on the default backend, or returns `nullptr` on failure.
StreamState * stream_create(uint32_t n_cores);
/// Creates a default-backend stream that prefers `core_ids`; an unavailable
/// exact set falls back to the `n_cores` request.
StreamState * stream_create_config(uint32_t n_cores, const int * core_ids, uint32_t core_id_count);
/// Creates a stream on `instance`, or returns `nullptr` on failure.
StreamState * stream_create_with_backend(BackendInstance * instance, uint32_t n_cores);
StreamState * stream_create_config_with_backend(BackendInstance * instance,
                                                uint32_t          n_cores,
                                                const int *       core_ids,
                                                uint32_t          core_id_count);
void          stream_destroy(StreamState *);
/// Returns the number of CC cores granted to `stream`, or zero if it is invalid.
uint32_t      stream_granted(const StreamState *);

// ---- launch / sync ----
// Launches `product(dims[0..ndim-1])` SPMD tiles that share one closure and are
// distinguished by program ID. The runtime assumes ownership of `closure` and
// invokes `del` exactly once, including failure paths after ownership transfer.
// Tiles are routed only to the stream's checked-out cores.
Future * launch_impl(StreamState *,
                     const uint32_t dims[3],
                     int            ndim,
                     TileInvoker    inv,
                     void *         closure,
                     ClosureDeleter del);
Future * launch_inline_impl(StreamState *,
                            const uint32_t dims[3],
                            int            ndim,
                            TileInvoker    inv,
                            void *         closure_source,
                            ClosureMover   mover,
                            ClosureDeleter inline_del);
Status   sync_impl(Future *);
/// Waits up to `timeout_ms` milliseconds without consuming the future handle.
Status   sync_timeout_impl(Future *, uint64_t timeout_ms);
/// Adds one owning reference to a future; a null pointer is ignored.
void     future_retain(Future *);
/// Releases one owning reference to a future; a null pointer is ignored.
void     future_release(Future *);

// ---- object creation (owned by the runtime arena) ----
// Barriers and events are bound to one stream. Handles use explicit
// retain/release operations and cross-stream coroutine waits are rejected.
Barrier * barrier_create(StreamState *, uint32_t total);
Event *   event_create(StreamState *);
void      barrier_retain(Barrier *);
void      barrier_release(Barrier *);
void      event_retain(Event *);
void      event_release(Event *);
Status    event_signal(Event *);
Status    event_reset(Event *);

// ---- context-scoped ops (called from inside a running tile) ----
// Every `Tile*` must identify the currently executing tile unless the operation
// explicitly accepts a null pointer. Returned views are non-owning.
uint32_t         ctx_program_id(const Tile *, int dim);
uint32_t         ctx_grid_dim(const Tile *, int dim);
Status           ctx_yield(Tile *);
Status           ctx_barrier(Tile *, Barrier *);
// Grid-wide barrier shortcut: synchronizes ALL tiles of the calling tile's
// launch (its whole grid) against the launch's implicit barrier. Equivalent to
// a ctx_barrier whose total is the grid product, but with no user-managed
// Barrier handle. Single-tile grids return Ok immediately.
Status           ctx_sync(Tile *);
Status           ctx_event_wait(Tile *, Event *);
Future *         ctx_spawn(Tile *, TileInvoker inv, void * closure, ClosureDeleter del);
Status           ctx_join(Tile *, Future *);
void             ctx_prefetch(const void * addr);
SharedBufferView ctx_shared_buffer();
SharedBufferView ctx_alloc_shared(size_t bytes, size_t alignment);
void             ctx_free_shared(SharedBufferView allocation);

const char * status_str(Status status);

}  // namespace spert::detail
