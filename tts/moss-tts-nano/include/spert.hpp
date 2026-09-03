// SPDX-FileCopyrightText: Copyright (c) 2026 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

// include/spert.hpp - Spert public C++ API (header-only facade).
//
// Spert: a Tile-SPMD many-core scheduling runtime for RISC-V. The public
// surface is modern C++: value/RAII types, `enum class Status`, and a
// variadic-template launch that perfectly forwards arbitrary kernel arguments.
//
// Kernel signature is user-defined; the first parameter is spert::Context*:
//
//   void matmul_tile(spert::Context* ctx, float* A, float* B, float* C,
//                    int64_t M, int64_t N, int64_t K,
//                    int64_t M_tile, int64_t N_tile);
//
//   spert::Stream s;   // checks out CC cores; runtime auto-inits on first use
//   auto fut = s.launch(spert::Grid{M / M_tile, N / N_tile},
//                       matmul_tile, A, B, C, M, N, K, M_tile, N_tile);
//   fut.sync();
//
// There is NO public Session: each backend handle maps to a hidden runtime
// instance with its own worker pool and arena. Users only deal with Streams;
// constructors without a BackendHandle retain the default-backend behavior.
//
// The arguments are captured by value (decayed) into a per-launch closure and
// reconstructed via std::apply when each tile runs: fn(ctx, args...).
#pragma once

#include "spert_engine.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace spert {

/// Returns a stable English name for a runtime status value.
///
/// The returned string has static storage duration and must not be freed.
inline const char * to_string(Status s) {
    return detail::status_str(s);
}

/// Describes the compute-core resources exposed by a backend.
///
/// All sizes and counts refer to CC (compute) cores rather than host CPU cores.
struct BackendInfo {
    size_t  shared_mem_size = 0;  ///< Shared-memory capacity available to each worker, in bytes.
    size_t  vlen            = 0;  ///< Architectural vector length reported by the backend, in bytes.
    size_t  num_cores       = 0;  ///< Number of CC cores exposed by the backend.
    int64_t core_arch_id    = 0;  ///< Backend-specific CC-core architecture identifier.
};

class BackendHandle;

/// Looks up or initializes a backend by name.
///
/// Returns an invalid handle when the name is empty, contains an embedded null
/// character, is unknown, or the backend cannot be initialized.
BackendHandle backend(std::string_view name);

/// Returns the lazily initialized default backend selected for this process.
///
/// The returned handle is invalid when backend selection or initialization
/// fails.
BackendHandle default_backend();

/// Returns information for `handle`, or a zero-initialized record when the
/// handle is invalid or incomplete.
BackendInfo backend_info(BackendHandle handle);

/// A non-owning identifier for a process-lifetime runtime backend instance.
///
/// Copies are inexpensive. The runtime owns the referenced instance and keeps
/// it alive until process shutdown.
class BackendHandle {
  public:
    BackendHandle() = default;

    /// Returns whether this handle refers to an initialized backend instance.
    bool valid() const { return inst_ != nullptr; }

    /// Returns the internal backend pointer, or `nullptr` for an invalid handle.
    ///
    /// This escape hatch is intended for integration code; ownership remains
    /// with the runtime.
    detail::BackendInstance * raw() const { return inst_; }

  private:
    explicit BackendHandle(detail::BackendInstance * instance) : inst_(instance) {}

    friend BackendHandle backend(std::string_view name);
    friend BackendHandle default_backend();

    detail::BackendInstance * inst_ = nullptr;
};

inline BackendHandle backend(std::string_view name) {
    if (name.empty() || name.find('\0') != std::string_view::npos) {
        return {};
    }
    const std::string owned_name(name);
    return BackendHandle(detail::backend_get(owned_name.c_str()));
}

inline BackendHandle default_backend() {
    return BackendHandle(detail::default_backend_instance());
}

/// Returns information for the process default backend.
///
/// A zero-initialized record indicates that the default backend is unavailable.
inline BackendInfo backend_info() {
    const detail::BackendInfo info = detail::backend_info();
    return { info.shared_mem_size, info.vlen, info.num_cores, info.core_arch_id };
}

inline BackendInfo backend_info(BackendHandle handle) {
    const detail::BackendInfo info = detail::backend_info(handle.raw());
    return { info.shared_mem_size, info.vlen, info.num_cores, info.core_arch_id };
}

/// Returns a point-in-time snapshot of resources retained by the default runtime.
///
/// Sampling takes cold-path registry locks. It does not add counters or atomic
/// operations to launch or worker hot paths.
inline RuntimeStats runtime_stats() {
    return detail::runtime_stats();
}

/// Describes a one-, two-, or three-dimensional SPMD launch grid.
///
/// The product of the active dimensions is the number of launched tiles.
/// Trailing dimensions are one and do not contribute additional tiles.
struct Grid {
    int      ndim    = 1;
    uint32_t dims[3] = { 1, 1, 1 };

    Grid() = default;

    explicit Grid(uint32_t x) : ndim(1), dims{ x, 1, 1 } {}

    Grid(uint32_t x, uint32_t y) : ndim(2), dims{ x, y, 1 } {}

    Grid(uint32_t x, uint32_t y, uint32_t z) : ndim(3), dims{ x, y, z } {}
};

// ---- forward decls ----
class Context;

// ============================================================
//  Type-erased closure machinery (perfect forwarding bridge)
// ============================================================
namespace detail {

// A closure that stores a decayed copy of the kernel and its arguments. When a
// tile runs, invoke() reconstructs fn(ctx, args...) via std::apply. The same
// closure instance is shared by all SPMD tiles of one launch; they differ only
// by Context (program_id).
template <typename Fn, typename... Args> struct TileClosure {
    Fn                  fn;
    std::tuple<Args...> args;

    template <typename F, typename... A>
    explicit TileClosure(F && f, A &&... a) : fn(std::forward<F>(f)), args(std::forward<A>(a)...) {}

    static void invoke(void * ctx, void * self_v) {
        auto * self = static_cast<TileClosure *>(self_v);
        std::apply([&](Args &... unpacked) { self->fn(reinterpret_cast<Context *>(ctx), unpacked...); }, self->args);
    }

    static void destroy(void * self_v) { delete static_cast<TileClosure *>(self_v); }

    static void move_inline(void * dst, void * src) noexcept {
        new (dst) TileClosure(std::move(*static_cast<TileClosure *>(src)));
    }

    static void destroy_inline(void * self_v) { static_cast<TileClosure *>(self_v)->~TileClosure(); }
};

// Build a heap closure from a kernel + forwarded args; decay everything so the
// closure owns its own copies (caller locals may go out of scope).
template <typename Fn, typename... Args>
TileClosure<std::decay_t<Fn>, std::decay_t<Args>...> * make_closure(Fn && fn, Args &&... args) {
    using C = TileClosure<std::decay_t<Fn>, std::decay_t<Args>...>;
    return new C(std::forward<Fn>(fn), std::forward<Args>(args)...);
}

}  // namespace detail

// ============================================================
//  Future
// ============================================================
/// A shared RAII handle to the completion state of one launch.
///
/// Copying retains the completion state, moving transfers a handle, and
/// destruction releases it. A default-constructed or moved-from future is
/// invalid.
class Future {
  public:
    Future() = default;

    Future(const Future & other) : fut_(other.fut_) { detail::future_retain(fut_); }

    Future(Future && other) noexcept : fut_(other.fut_) { other.fut_ = nullptr; }

    Future & operator=(const Future & other) {
        if (this != &other) {
            detail::future_retain(other.fut_);
            detail::future_release(fut_);
            fut_ = other.fut_;
        }
        return *this;
    }

    Future & operator=(Future && other) noexcept {
        if (this != &other) {
            detail::future_release(fut_);
            fut_       = other.fut_;
            other.fut_ = nullptr;
        }
        return *this;
    }

    ~Future() { detail::future_release(fut_); }

    /// Waits until the launch reaches a terminal state.
    ///
    /// This overload uses hot polling and has no timeout. It returns
    /// `Status::InvalidArg` for an invalid future.
    Status sync() const { return detail::sync_impl(fut_); }

    /// Waits for at most `timeout_ms` milliseconds for terminal completion.
    ///
    /// Returns `Status::Timeout` without consuming or invalidating the future
    /// when the deadline expires.
    Status sync(uint64_t timeout_ms) const { return detail::sync_timeout_impl(fut_, timeout_ms); }

    /// Waits for every future and returns the first non-OK terminal status.
    static Status sync_all(std::initializer_list<Future> futures) {
        Status first = Status::Ok;
        for (const Future & f : futures) {
            Status st = f.sync();
            if (first == Status::Ok && st != Status::Ok) {
                first = st;
            }
        }
        return first;
    }

    /// Waits for every future and returns the first non-OK terminal status.
    static Status sync_all(const std::vector<Future> & futures) {
        Status first = Status::Ok;
        for (const Future & f : futures) {
            Status st = f.sync();
            if (first == Status::Ok && st != Status::Ok) {
                first = st;
            }
        }
        return first;
    }

    /// Waits for all futures under one shared timeout budget in milliseconds.
    static Status sync_all(std::initializer_list<Future> futures, uint64_t timeout_ms) {
        return sync_all_timeout(futures.begin(), futures.end(), timeout_ms);
    }

    /// Waits for all futures under one shared timeout budget in milliseconds.
    static Status sync_all(const std::vector<Future> & futures, uint64_t timeout_ms) {
        return sync_all_timeout(futures.begin(), futures.end(), timeout_ms);
    }

    /// Returns whether this handle refers to a launch completion state.
    bool valid() const { return fut_ != nullptr; }

    /// Returns the internal completion pointer without transferring ownership.
    detail::Future * raw() const { return fut_; }

  private:
    friend class Stream;
    friend class Context;

    // Adopts the initial reference returned by launch/spawn. Keeping this private
    // prevents callers from manufacturing a second, unretained owner via raw().
    explicit Future(detail::Future * f) : fut_(f) {}

    template <typename Iterator> static Status sync_all_timeout(Iterator begin, Iterator end, uint64_t timeout_ms) {
        for (Iterator it = begin; it != end; ++it) {
            if (!it->valid()) {
                return Status::InvalidArg;
            }
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        Status     first    = Status::Ok;
        for (Iterator it = begin; it != end; ++it) {
            const auto now          = std::chrono::steady_clock::now();
            uint64_t   remaining_ms = 0;
            if (now < deadline) {
                const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
                remaining_ms         = static_cast<uint64_t>(remaining > 0 ? remaining : 1);
            }
            const Status status = it->sync(remaining_ms);
            if (status == Status::Timeout) {
                return status;
            }
            if (first == Status::Ok && status != Status::Ok) {
                first = status;
            }
        }
        return first;
    }

    detail::Future * fut_ = nullptr;
};

// ============================================================
//  Barrier / Event (handles; owned by the session arena)
// ============================================================
/// A shared RAII handle to a reusable, stream-scoped barrier.
///
/// The barrier may be used only by tiles belonging to the stream that created
/// it. Cross-stream use returns `Status::CrossStream`.
class Barrier {
  public:
    Barrier() = default;

    Barrier(const Barrier & other) : b_(other.b_) { detail::barrier_retain(b_); }

    Barrier(Barrier && other) noexcept : b_(other.b_) { other.b_ = nullptr; }

    Barrier & operator=(const Barrier & other) {
        if (this != &other) {
            detail::barrier_retain(other.b_);
            detail::barrier_release(b_);
            b_ = other.b_;
        }
        return *this;
    }

    Barrier & operator=(Barrier && other) noexcept {
        if (this != &other) {
            detail::barrier_release(b_);
            b_       = other.b_;
            other.b_ = nullptr;
        }
        return *this;
    }

    ~Barrier() { detail::barrier_release(b_); }

    /// Returns the internal barrier pointer without transferring ownership.
    detail::Barrier * raw() const { return b_; }

    /// Returns whether this handle refers to a runtime barrier.
    bool valid() const { return b_ != nullptr; }

  private:
    friend class Stream;

    // Adopts the initial reference returned by make_barrier(). Keeping this
    // private prevents callers from forging an unretained owner via raw().
    explicit Barrier(detail::Barrier * b) : b_(b) {}

    detail::Barrier * b_ = nullptr;
};

/// A shared RAII handle to a manual-reset, stream-scoped event.
///
/// Signaling wakes current waiters and leaves the event signaled until
/// `reset()` is called. Tiles from other streams may not wait on this event.
class Event {
  public:
    Event() = default;

    Event(const Event & other) : e_(other.e_) { detail::event_retain(e_); }

    Event(Event && other) noexcept : e_(other.e_) { other.e_ = nullptr; }

    Event & operator=(const Event & other) {
        if (this != &other) {
            detail::event_retain(other.e_);
            detail::event_release(e_);
            e_ = other.e_;
        }
        return *this;
    }

    Event & operator=(Event && other) noexcept {
        if (this != &other) {
            detail::event_release(e_);
            e_       = other.e_;
            other.e_ = nullptr;
        }
        return *this;
    }

    ~Event() { detail::event_release(e_); }

    /// Signals the event and wakes its current waiters.
    Status signal() { return detail::event_signal(e_); }

    /// Returns the event to the non-signaled state.
    Status reset() { return detail::event_reset(e_); }

    /// Returns the internal event pointer without transferring ownership.
    detail::Event * raw() const { return e_; }

    /// Returns whether this handle refers to a runtime event.
    bool valid() const { return e_ != nullptr; }

  private:
    friend class Stream;

    // Adopts the initial reference returned by make_event(). Keeping this
    // private prevents callers from forging an unretained owner via raw().
    explicit Event(detail::Event * e) : e_(e) {}

    detail::Event * e_ = nullptr;
};

// ============================================================
//  Context (passed to every kernel; opaque handle to a running tile)
// ============================================================
/// The execution context supplied to each running SPMD tile.
///
/// Context objects are created and owned by the runtime. A context is valid
/// only for the duration of its kernel invocation and must not be retained.
class Context {
  public:
    /// Returns this tile's zero-based coordinate on `dim`, or zero for an
    /// invalid dimension.
    uint32_t program_id(int dim = 0) const { return detail::ctx_program_id(tile(), dim); }

    /// Returns the extent of the launch grid on `dim`, or zero for an invalid
    /// dimension.
    uint32_t grid_dim(int dim = 0) const { return detail::ctx_grid_dim(tile(), dim); }

    /// Cooperatively yields execution so another ready tile may run.
    Status yield() { return detail::ctx_yield(tile()); }

    /// Waits on a barrier created by the same stream.
    Status barrier(Barrier & b) { return detail::ctx_barrier(tile(), b.raw()); }

    /// Waits until every tile in this launch reaches the same grid barrier.
    ///
    /// This is equivalent to a barrier whose total is the grid product, but it
    /// requires no user-managed handle. A single-tile grid returns immediately.
    Status sync() { return detail::ctx_sync(tile()); }

    /// Waits for a stream-scoped event to become signaled.
    Status wait(Event & e) { return detail::ctx_event_wait(tile(), e.raw()); }

    /// Spawns one child tile and returns an owning future for its completion.
    ///
    /// The callable and arguments are captured by value. Use `join()` from the
    /// same stream to wait cooperatively for the returned future.
    template <typename Fn, typename... Args> Future spawn(Fn && fn, Args &&... args);

    /// Cooperatively waits for a future created on the same stream.
    Status join(Future & f) { return detail::ctx_join(tile(), f.raw()); }

    /// Issues a best-effort read prefetch for `addr`.
    void prefetch(const void * addr) { detail::ctx_prefetch(addr); }

    /// Returns the entire shared scratch buffer of the current worker.
    ///
    /// The view is non-owning and remains valid only while this tile executes on
    /// that worker.
    SharedBufferView shared_buffer() const { return detail::ctx_shared_buffer(); }

    /// Allocates `bytes` from the current worker's shared-memory pool.
    ///
    /// The call may suspend cooperatively until space becomes available. An
    /// empty view indicates invalid input or allocation failure. The allocation
    /// must be released with `free_shared()` by the same tile. `alignment` is
    /// reserved for allocator selection; the current pool uses its fixed block
    /// alignment.
    SharedBufferView alloc_shared(size_t bytes, size_t alignment = 64) {
        return detail::ctx_alloc_shared(bytes, alignment);
    }

    /// Releases an allocation previously returned by `alloc_shared()`.
    void free_shared(SharedBufferView allocation) { detail::ctx_free_shared(allocation); }

  private:
    // Context is never constructed by users; the engine reinterprets a
    // detail::Tile* as Context*. tile() recovers it.
    detail::Tile * tile() { return reinterpret_cast<detail::Tile *>(this); }

    const detail::Tile * tile() const { return reinterpret_cast<const detail::Tile *>(this); }
};

// ============================================================
//  Stream (RAII owner of a checked-out CC-core subset)
// ============================================================
// A Stream checks out a subset of the CC cores at construction (FIFO-fair via
// the SHM arbiter) and routes its launched tiles ONLY to workers bound to those
// cores -> different streams run on disjoint physical cores in true parallel.
// Barriers/events are stream-scoped (no cross-stream coroutine waits).
//
// Creating the first Stream for a backend lazily brings up that backend's
// worker pool and engine arena. There is no user-visible Session.
/// Configures the CC-core grant requested by a stream.
struct StreamConfig {
    uint32_t         n_cores = 0;  ///< Requested core count; zero requests all available CC cores.
    /// Preferred physical CC-core identifiers. If the exact set cannot be
    /// granted, stream creation falls back to the `n_cores` request.
    std::vector<int> core_ids;
};

/// A move-only RAII owner of a stream-scoped CC-core grant.
///
/// A stream routes its launches only to workers in its grant. Destroying the
/// stream cancels outstanding work, marks surviving stream-scoped handles as
/// cancelled, and returns the grant to its backend.
class Stream {
  public:
    /// Creates a stream on the default backend using `cfg`.
    explicit Stream(const StreamConfig & cfg = {}) : Stream(default_backend(), cfg) {}

    /// Creates a stream on the default backend requesting `n_cores` CC cores.
    /// Zero requests all available cores.
    explicit Stream(uint32_t n_cores) : Stream(default_backend(), n_cores) {}

    /// Creates a stream on `handle` and requests all available CC cores.
    explicit Stream(BackendHandle handle) : Stream(handle, StreamConfig{}) {}

    /// Creates a stream on `handle` using `cfg`.
    ///
    /// The resulting stream is invalid if the backend is invalid, shutting
    /// down, or cannot grant the request.
    Stream(BackendHandle handle, const StreamConfig & cfg) {
        st_ = detail::stream_create_config_with_backend(handle.raw(), cfg.n_cores, cfg.core_ids.data(),
                                                        (uint32_t) cfg.core_ids.size());
    }

    /// Creates a stream on `handle` requesting `n_cores` CC cores.
    Stream(BackendHandle handle, uint32_t n_cores) { st_ = detail::stream_create_with_backend(handle.raw(), n_cores); }

    ~Stream() {
        if (st_) {
            detail::stream_destroy(st_);
        }
    }

    Stream(const Stream &)             = delete;
    Stream & operator=(const Stream &) = delete;

    Stream(Stream && o) noexcept : st_(o.st_) { o.st_ = nullptr; }

    Stream & operator=(Stream && o) noexcept {
        if (this != &o) {
            if (st_) {
                detail::stream_destroy(st_);
            }
            st_   = o.st_;
            o.st_ = nullptr;
        }
        return *this;
    }

    /// Returns whether stream construction acquired a runtime stream state.
    bool valid() const { return st_ != nullptr; }

    /// Returns the number of CC cores granted to this stream.
    uint32_t core_count() const { return st_ ? detail::stream_granted(st_) : 0; }

    /// Launches one SPMD tile per element of `g` on this stream's core grant.
    ///
    /// The callable and arguments are decay-copied into launch-owned storage.
    /// The returned future is invalid if the launch cannot be admitted.
    template <typename Fn, typename... Args> Future launch(const Grid & g, Fn && fn, Args &&... args) {
        using C = detail::TileClosure<std::decay_t<Fn>, std::decay_t<Args>...>;
        if constexpr (sizeof(C) <= detail::kInlineClosureBytes && alignof(C) <= alignof(std::max_align_t) &&
                      std::is_nothrow_move_constructible_v<C>) {
            C closure(std::forward<Fn>(fn), std::forward<Args>(args)...);
            return Future(detail::launch_inline_impl(st_, g.dims, g.ndim, &C::invoke, &closure, &C::move_inline,
                                                     &C::destroy_inline));
        } else {
            auto * closure = detail::make_closure(std::forward<Fn>(fn), std::forward<Args>(args)...);
            return Future(detail::launch_impl(st_, g.dims, g.ndim, &C::invoke, closure, &C::destroy));
        }
    }

    /// Creates a stream-scoped barrier for `total` participants.
    /// A zero total or invalid stream produces an invalid handle.
    Barrier make_barrier(uint32_t total) { return Barrier(detail::barrier_create(st_, total)); }

    /// Creates a manual-reset event scoped to this stream.
    Event make_event() { return Event(detail::event_create(st_)); }

    /// Returns the internal stream pointer without transferring ownership.
    detail::StreamState * raw() const { return st_; }

  private:
    detail::StreamState * st_ = nullptr;
};

// ---- Context::spawn (defined after Stream for completeness) ----
template <typename Fn, typename... Args> Future Context::spawn(Fn && fn, Args &&... args) {
    auto * closure     = detail::make_closure(std::forward<Fn>(fn), std::forward<Args>(args)...);
    using C            = std::remove_pointer_t<decltype(closure)>;
    detail::Future * f = detail::ctx_spawn(tile(), &C::invoke, closure, &C::destroy);
    return Future(f);
}

}  // namespace spert
