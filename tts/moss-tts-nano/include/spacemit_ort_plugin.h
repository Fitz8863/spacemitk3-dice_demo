// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Forward declare EP-internal bridge types used for friendship.
namespace onnxruntime {
namespace spacemit {
class PluginOperatorBridge;
class PluginCapabilityBridge;
class PluginRegistry;
}  // namespace spacemit
}  // namespace onnxruntime

namespace spacemit {
namespace plugin {

// ============================================================================
// DataType — aligned with ONNX TensorProto_DataType
// ============================================================================

enum class DataType : int32_t {
    UNDEFINED  = 0,
    FLOAT32    = 1,
    UINT8      = 2,
    INT8       = 3,
    UINT16     = 4,
    INT16      = 5,
    INT32      = 6,
    INT64      = 7,
    STRING     = 8,
    BOOL       = 9,
    FLOAT16    = 10,
    DOUBLE     = 11,
    UINT32     = 12,
    UINT64     = 13,
    COMPLEX64  = 14,
    COMPLEX128 = 15,
    BFLOAT16   = 16,
};

// ============================================================================
// Utility functions (provided by the plugin framework)
// ============================================================================

/// Return the byte size for a given DataType.
/// Unknown types return 0; callers should handle that case.
inline size_t DataTypeByteSize(DataType dt) noexcept {
    switch (dt) {
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::COMPLEX64:
            return 4;
        case DataType::FLOAT16:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::BFLOAT16:
            return 2;
        case DataType::INT64:
        case DataType::UINT64:
        case DataType::DOUBLE:
        case DataType::COMPLEX128:
            return 8;
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::BOOL:
        case DataType::STRING:
            return 1;
        default:
            return 0;
    }
}

// ============================================================================
// Forward declarations
// ============================================================================

class SpinePluginNode;
class SpinePluginContext;
class SpinePluginTensor;
class SpinePluginGraphViewer;
class SpineCustomDispatch;

// ============================================================================
// SpinePluginTensor — tensor wrapper for plugin I/O
// ============================================================================

class SpinePluginTensor {
  public:
    DataType                     GetDataType() const;
    const std::vector<int64_t> & GetShape() const;
    const void *                 GetData() const;
    void *                       GetMutableData();
    size_t                       GetElementCount() const;
    size_t                       GetByteSize() const;

    void * internal_ptr_ = nullptr;
};

// ============================================================================
// SpinePluginNode — node info (op type, domain, attributes, shapes)
// ============================================================================

class SpinePluginNode {
  public:
    const std::string & GetOpType() const;
    const std::string & GetDomain() const;
    const std::string & GetName() const;

    size_t GetInputCount() const;
    size_t GetOutputCount() const;

    DataType GetInputType(size_t index) const;
    DataType GetOutputType(size_t index) const;

    std::vector<int64_t> GetInputShape(size_t index) const;
    std::vector<int64_t> GetOutputShape(size_t index) const;

    // Attribute readers
    float                GetAttributeFloat(const std::string & name, float default_value = 0.0f) const;
    int64_t              GetAttributeInt(const std::string & name, int64_t default_value = 0) const;
    std::string          GetAttributeString(const std::string & name, const std::string & default_value = "") const;
    std::vector<int64_t> GetAttributeInts(const std::string & name) const;
    std::vector<float>   GetAttributeFloats(const std::string & name) const;

    void * internal_ptr_ = nullptr;
};

// ============================================================================
// SpinePluginGraphViewer — graph-level queries for capability checks
// ============================================================================

class SpinePluginGraphViewer {
  public:
    size_t                  GetNodeCount() const;
    const SpinePluginNode * GetNode(size_t index) const;

    void * internal_ptr_ = nullptr;
};

// ============================================================================
// SpinePluginContext — per-operator call context
// ============================================================================

class SpinePluginContext {
  public:
    const SpinePluginNode & GetNode() const;

    const SpinePluginTensor & GetInputTensor(size_t index) const;
    SpinePluginTensor &       GetOutputTensor(size_t index);

    void SetOutputTensorInfo(size_t index, DataType dtype, const std::vector<int64_t> & shape);

    size_t GetInputCount() const;
    size_t GetOutputCount() const;

    /// Thread index (0..tnum-1) for EP-managed parallelism.
    ptrdiff_t GetThreadIndex() const;
    /// Total number of threads in this parallel group.
    ptrdiff_t GetThreadCount() const;
    /// Optional thread ID offset for nested parallelism.
    ptrdiff_t GetThreadOffset() const;

    /// Install a custom compute dispatch on the current operator.
    /// Takes ownership of dispatch when it is accepted. Call from KernelDispatch().
    void SetCustomDispatch(SpineCustomDispatch * dispatch);

    /// Returns true when this context is backed by an EP-internal operator.
    /// Direct unit-test/default-constructed contexts return false.
    bool IsValid() const noexcept { return internal_ptr_ != nullptr; }

  private:
    void * internal_ptr_ = nullptr;
    friend class ::onnxruntime::spacemit::PluginOperatorBridge;
    friend class ::onnxruntime::spacemit::PluginCapabilityBridge;
};

// ============================================================================
// SpineCustomDispatch — custom compute dispatch base class.
//
// Subclass this and implement operator()() to provide the operator's
// computation. KernelDispatch() typically creates a dispatch with new, checks
// Support(), then passes ownership to SpinePluginContext::SetCustomDispatch().
// The framework calls operator()() for each thread grain assigned by the EP
// scheduler.
//
// Dispatch() is called immediately by SetCustomDispatch() for setup.
// ReDispatch() is called by the EP on shape changes. Support() lets plugin
// code decide whether to install this dispatch.
//
// internal_ptr_ is set by the EP after KernelDispatch to the internal
// SpineComputeDispatch; available for advanced dispatch state inspection.
// ============================================================================

class SpineCustomOperator;

class SpineCustomDispatch {
  public:
    virtual ~SpineCustomDispatch() = default;

    /// Called by the EP during dispatch setup phase. Default: no-op.
    virtual void Dispatch(SpinePluginContext & ctx) {}

    /// Called by the EP during re-dispatch (shape change) phase. Default: no-op.
    virtual void ReDispatch(SpinePluginContext & ctx) {}

    /// Return whether this dispatch supports the current operator state. Default: supported.
    virtual bool Support(SpinePluginContext & ctx) {
        (void) ctx;
        return true;
    }

    /// Execute one thread grain. The context provides thread index/count
    /// and input/output tensor access.
    virtual void operator()(SpinePluginContext & ctx) = 0;

    /// Set by the EP to the internal SpineComputeDispatch after KernelDispatch.
    void * internal_ptr_ = nullptr;
};

// ============================================================================
// PluginOperatorKind — determines which interface the plugin must implement
// ============================================================================

enum class PluginOperatorKind {
    /// registrar->AddOperator: full operator + dispatch (Track 1).
    /// Op type must NOT conflict with EP built-in types.
    /// Required methods: Compile, ReCompile, KernelDispatch, CheckCapability, Predict.
    FullOperator,

    /// registrar->AddDispatch: dispatch-only overlay (Track 2).
    /// Op type MUST match an existing EP built-in type.
    /// Required methods: Compile, KernelDispatch, Predict.
    /// No ReCompile (no shape inference), no SetOutputTensorInfo.
    DispatchOnly,
};

// ============================================================================
// SpineCustomOperator — base class for all plugin operators.
//
// Track 1 (FullOperator / AddOperator):
//   Must implement: Compile, ReCompile, KernelDispatch, Predict, CheckCapability.
//
// Track 2 (DispatchOnly / AddDispatch):
//   Must implement: Compile, KernelDispatch, Predict.
//   ReCompile is not called (the EP built-in operator handles shape inference).
//   KernelDispatch must install a compute dispatch via SetComputeDispatch on
//   the bridge, or implement Predict() for single-threaded execution.
// ============================================================================

class SpineCustomOperator {
  public:
    virtual ~SpineCustomOperator() = default;

    /// Compile: read node attributes, initialize operator state.
    /// Called once when the operator instance is created.
    virtual void Compile(SpinePluginContext & ctx) = 0;

    /// ReCompile: handle dynamic shape changes and set output tensor info.
    /// Called on every shape change. MUST call ctx.SetOutputTensorInfo() for
    /// each output. DispatchOnly plugins may leave this empty — the EP built-in
    /// operator handles shape inference for them.
    virtual void ReCompile(SpinePluginContext & ctx) { /* DispatchOnly: no-op */ }

    /// KernelDispatch: set the compute strategy (threading, tiling).
    /// Called after ReCompile (FullOperator) or after Compile (DispatchOnly).
    /// MUST be implemented — even for simple ops that delegate to Predict().
    virtual void KernelDispatch(SpinePluginContext & ctx) = 0;

    /// Predict: execute the actual computation.
    /// Called on every inference run. Uses the dispatch strategy set by
    /// KernelDispatch. The default implementation delegates to the dispatch.
    virtual void Predict(SpinePluginContext & ctx) {
        // Default: uses the dispatch set by KernelDispatch.
    }

    /// CheckCapability: determine whether this plugin supports the given node.
    /// Returns {supported, confidence_score 0-100}.
    /// Default: supports all nodes with score 100.
    virtual std::pair<bool, int64_t> CheckCapability(const SpinePluginNode &        node,
                                                     const SpinePluginGraphViewer & graph_viewer) const {
        return { true, 100 };
    }
};

// ============================================================================
// PluginRegistrar — fluent builder for registering operators with the EP
// ============================================================================

class PluginRegistrar;  // Forward declaration; injected by the EP plugin loader.

/// PluginRegistrar: fluent builder for batch operator registration.
class PluginRegistrar {
  public:
    PluginRegistrar() = default;
    ~PluginRegistrar();
    PluginRegistrar(const PluginRegistrar &)             = delete;
    PluginRegistrar & operator=(const PluginRegistrar &) = delete;

    // ── Legacy Add(): backward-compatible single-track registration ──
    template <typename OpClass> PluginRegistrar & Add(const std::string & op_type, const std::string & domain = "") {
        AddImpl(
            op_type, domain, PluginOperatorKind::FullOperator,
            []() -> std::unique_ptr<SpineCustomOperator> { return std::make_unique<OpClass>(); },
            [](const SpinePluginNode & node, const SpinePluginGraphViewer & viewer) {
                OpClass tmp;
                return tmp.CheckCapability(node, viewer);
            });
        return *this;
    }

    // ── AddOperator: full operator + dispatch (Track 1) ──
    // Op type MUST NOT conflict with EP built-in types.
    // Class must implement: Compile, ReCompile, KernelDispatch, CheckCapability, Predict.
    template <typename OpClass>
    PluginRegistrar & AddOperator(const std::string & op_type, const std::string & domain = "") {
        AddImpl(
            op_type, domain, PluginOperatorKind::FullOperator,
            []() -> std::unique_ptr<SpineCustomOperator> { return std::make_unique<OpClass>(); },
            [](const SpinePluginNode & node, const SpinePluginGraphViewer & viewer) {
                OpClass tmp;
                return tmp.CheckCapability(node, viewer);
            });
        return *this;
    }

    // ── AddDispatch: dispatch-only overlay (Track 2) ──
    // Op type MUST match an existing EP built-in type.
    // Class must implement: Compile, KernelDispatch, Predict.
    // No ReCompile (no shape inference), no SetOutputTensorInfo.
    template <typename OpClass>
    PluginRegistrar & AddDispatch(const std::string & op_type, const std::string & domain = "") {
        AddImpl(
            op_type, domain, PluginOperatorKind::DispatchOnly,
            []() -> std::unique_ptr<SpineCustomOperator> { return std::make_unique<OpClass>(); },
            [](const SpinePluginNode & node, const SpinePluginGraphViewer & viewer) {
                OpClass tmp;
                return tmp.CheckCapability(node, viewer);
            });
        return *this;
    }

    size_t Count() const noexcept;

  private:
    // INTERNAL USE ONLY — called by the EP plugin loader.
    // User plugins should use Add<T>() / AddOperator<T>() / AddDispatch<T>().
    using FactoryFn = std::function<std::unique_ptr<SpineCustomOperator>()>;
    using CheckerFn = std::function<std::pair<bool, int64_t>(const SpinePluginNode &, const SpinePluginGraphViewer &)>;
    void AddImpl(const std::string & op_type,
                 const std::string & domain,
                 PluginOperatorKind  kind,
                 FactoryFn           factory,
                 CheckerFn           checker);

    void * internal_ptr_ = nullptr;

  private:
    // EP-internal: set the registrar context pointer.
    void _SetContext(void * ctx) { internal_ptr_ = ctx; }
    friend class ::onnxruntime::spacemit::PluginRegistry;
};

// ============================================================================
// Declarative batch registration macros
// ============================================================================

namespace detail {
template <typename...> struct OpList {};

template <typename First, typename... Rest>
inline void AddEach(PluginRegistrar & r, const std::string & domain, OpList<First, Rest...>) {
    r.Add<First>(/*op_type=*/"", domain);  // WIP: op_type deduction from First::kOpType
    if constexpr (sizeof...(Rest) > 0) {
        AddEach(r, domain, OpList<Rest...>{});
    }
}

inline void AddEach(PluginRegistrar &, const std::string &, OpList<>) {}
}  // namespace detail

#define SPACEMIT_PLUGIN_REGISTER_OP(registrar, domain, OpClass, op_type) \
    do {                                                                 \
        (registrar).Add<OpClass>(op_type, domain);                       \
    } while (0)

// ============================================================================
// Plugin metadata and version declaration
// ============================================================================

#define SPACEMIT_PLUGIN_ABI_VERSION 1u

struct PluginMetadata {
    const char * name;
    const char * version;
    const char * min_ep_version;
    const char * description;
    const char * author;
};

/// SPACEMIT_PLUGIN_DECLARE_VERSION(name, version, min_ep_version, desc, author)
///
/// Call once in the plugin source file. Automatically generates the two
/// extern "C" entry points required by the EP loader:
///   - SpacemitPluginGetAbiVersion()
///   - SpacemitPluginGetMetadata()
///
/// Usage:
///   SPACEMIT_PLUGIN_DECLARE_VERSION("MyPlugin", "1.0.0", "2.0.0",
///                                   "My custom plugin", "Author Name");
#define SPACEMIT_PLUGIN_DECLARE_VERSION(plugin_name, plugin_version, ep_min_version, ...)                         \
    extern "C" __attribute__((visibility("default"))) uint32_t SpacemitPluginGetAbiVersion() {                    \
        return SPACEMIT_PLUGIN_ABI_VERSION;                                                                       \
    }                                                                                                             \
    extern "C" __attribute__((visibility("default"))) const ::spacemit::plugin::PluginMetadata *                  \
    SpacemitPluginGetMetadata() {                                                                                 \
        static const ::spacemit::plugin::PluginMetadata kMeta{ (plugin_name), (plugin_version), (ep_min_version), \
                                                               ##__VA_ARGS__ };                                   \
        return &kMeta;                                                                                            \
    }

}  // namespace plugin
}  // namespace spacemit

// ============================================================================
// Plugin entry point — must be implemented by every plugin
// ============================================================================

extern "C" void SpacemitPluginInit(::spacemit::plugin::PluginRegistrar * registrar);
