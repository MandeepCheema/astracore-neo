// SPDX-License-Identifier: Apache-2.0
//
// AstraCore Neo — C++ runtime public header (v0.1).
//
// What this is for
// ----------------
// OEM firmware deploys in C/C++, not Python. This header is the
// surface a customer-side build links against to drive a compiled
// model on whatever target silicon the customer chooses. The same
// API is exposed to Python via pybind11 (see ``cpp/python/bindings.cpp``)
// so the Python SDK and the C++ deployment artefact share one
// implementation.
//
// Status
// ------
// v0.1 ships:
//   * ``Backend`` interface
//   * ``OrtBackend`` reference implementation backed by ONNX Runtime
//     (the same EP set the Python OrtBackend exposes — CPU today,
//     CUDA / TensorRT / OpenVINO / QNN once the matching ORT build is
//     installed; see docs/cloud_readiness_playbook.md).
//   * ``Tensor`` lightweight wrapper for input / output buffers.
//
// Future (Phase B follow-on):
//   * Direct silicon backends (TensorRT plan loader, SNPE / QNN, F1 XRT)
//   * Stable ABI guarantee for cross-compiler linking
//   * Quantisation API (currently the Python `astracore quantise`
//     emits an INT8 ONNX, which OrtBackend then loads transparently)

#ifndef ASTRACORE_RUNTIME_HPP_
#define ASTRACORE_RUNTIME_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace astracore {

// ---------------------------------------------------------------------------
// Tensor — a typed multi-dim buffer that the runtime borrows.
//
// Ownership: the caller owns the data buffer. The Tensor is a non-
// owning view (`data` pointer + shape + dtype). Output tensors
// returned by ``Backend::run`` own their buffers via shared_ptr.
// ---------------------------------------------------------------------------

enum class DType : int {
  kFloat32 = 1,
  kUInt8   = 2,
  kInt8    = 3,
  kInt16   = 5,
  kInt32   = 6,
  kInt64   = 7,
  kBool    = 9,
  kFloat16 = 10,
  kFloat64 = 11,
};

class Tensor {
 public:
  Tensor() = default;
  Tensor(DType dtype, std::vector<int64_t> shape, void* data,
         std::size_t bytes)
      : dtype_(dtype), shape_(std::move(shape)),
        data_(data), bytes_(bytes), owned_(nullptr) {}

  // Owning ctor — used for output tensors returned by run().
  Tensor(DType dtype, std::vector<int64_t> shape,
         std::shared_ptr<std::uint8_t[]> owned, std::size_t bytes)
      : dtype_(dtype), shape_(std::move(shape)),
        data_(owned.get()), bytes_(bytes), owned_(std::move(owned)) {}

  DType dtype() const noexcept { return dtype_; }
  const std::vector<int64_t>& shape() const noexcept { return shape_; }
  void* data() noexcept { return data_; }
  const void* data() const noexcept { return data_; }
  std::size_t bytes() const noexcept { return bytes_; }
  int64_t element_count() const noexcept {
    int64_t n = 1;
    for (auto d : shape_) n *= d;
    return n;
  }

 private:
  DType dtype_ = DType::kFloat32;
  std::vector<int64_t> shape_;
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
  std::shared_ptr<std::uint8_t[]> owned_;
};

// ---------------------------------------------------------------------------
// Report — same KPI surface as the Python BackendReport.
// ---------------------------------------------------------------------------

struct Report {
  std::string backend;            // "onnxruntime"
  std::string model;              // path basename
  std::vector<std::string> active_providers;  // EPs ORT actually used
  int    n_inferences = 0;
  double wall_s_total = 0.0;
  double wall_ms_per_inference = 0.0;
  std::int64_t mac_ops_total = 0;
  double delivered_tops = 0.0;
};

// ---------------------------------------------------------------------------
// Backend — the contract every silicon target implements.
// ---------------------------------------------------------------------------

// Opaque base — backends derive concrete types holding their own
// session / engine handles. Defined here (not forward-declared) so
// std::unique_ptr<Program> can call the destructor without the impl.
class Program {
 public:
  virtual ~Program() = default;
};

class Backend {
 public:
  virtual ~Backend() = default;

  // Compile an ONNX model file path into a runnable program.
  virtual std::unique_ptr<Program> compile(const std::string& onnx_path) = 0;

  // Execute one inference. Returns name -> output tensor.
  virtual std::map<std::string, Tensor> run(
      Program& program,
      const std::map<std::string, Tensor>& inputs) = 0;

  // KPIs from the most recent run.
  virtual const Report& report_last() const = 0;

  virtual const std::string& name() const = 0;
};

// Provider spec — either a bare name (short alias OK) or a
// (name, options) pair. Matches the Python OrtBackend shape.
struct ProviderSpec {
  std::string name;
  std::map<std::string, std::string> options;   // kebab-case ORT EP keys
  // Convenience ctors so callers can pass either just a string or a
  // (string, {{"k","v"}, ...}) struct literal.
  ProviderSpec(const char* n) : name(n) {}
  ProviderSpec(const std::string& n) : name(n) {}
  ProviderSpec(std::string n,
               std::map<std::string, std::string> o)
      : name(std::move(n)), options(std::move(o)) {}
};

// Factory — picks the right backend by name.
//
// v0.1 signature kept for compatibility; v0.2 adds the ProviderSpec
// overload so callers can pass per-EP options (TRT workspace size,
// CUDA device_id, OpenVINO device_type, QNN backend_path, ...).
std::unique_ptr<Backend> make_backend(
    const std::string& name,
    const std::vector<std::string>& providers = {});

std::unique_ptr<Backend> make_backend_with_options(
    const std::string& name,
    const std::vector<ProviderSpec>& providers);

// Library version — semantic versioning string ("0.1.0").
const char* version();

}  // namespace astracore

#endif  // ASTRACORE_RUNTIME_HPP_
