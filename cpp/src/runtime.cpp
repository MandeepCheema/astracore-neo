// SPDX-License-Identifier: Apache-2.0
//
// AstraCore Neo — C++ runtime implementation (v0.1).
//
// Wraps ONNX Runtime via its C++ API. Same EP plumbing the Python
// OrtBackend uses, exposed through the astracore::Backend interface.

#include "astracore/runtime.hpp"

// Vendored ORT headers — see cpp/third_party/onnxruntime/.
#include "onnxruntime_cxx_api.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace astracore {

namespace {

// ---- ONNX type → astracore::DType bridge -------------------------------

DType from_ort(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:    return DType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:    return DType::kUInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:     return DType::kInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:    return DType::kInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:    return DType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:    return DType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:     return DType::kBool;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:  return DType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:   return DType::kFloat64;
    default: return DType::kFloat32;
  }
}

ONNXTensorElementDataType to_ort(DType t) {
  switch (t) {
    case DType::kFloat32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DType::kUInt8:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DType::kInt8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DType::kInt16:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case DType::kInt32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DType::kInt64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DType::kBool:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case DType::kFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DType::kFloat64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

std::size_t element_size(DType t) {
  switch (t) {
    case DType::kFloat32: case DType::kInt32:                  return 4;
    case DType::kUInt8:   case DType::kInt8:  case DType::kBool: return 1;
    case DType::kInt16:   case DType::kFloat16:                  return 2;
    case DType::kInt64:   case DType::kFloat64:                  return 8;
  }
  return 4;
}

}  // namespace

// ---------------------------------------------------------------------------
// OrtProgram — concrete Program owning an ORT InferenceSession.
// ---------------------------------------------------------------------------

class OrtProgram : public Program {
 public:
  Ort::Session session{nullptr};
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

// ---------------------------------------------------------------------------
// OrtBackend — wraps Ort::Env + Ort::Session.
// ---------------------------------------------------------------------------

class OrtBackend : public Backend {
 public:
  explicit OrtBackend(std::vector<std::string> providers)
      : env_(ORT_LOGGING_LEVEL_WARNING, "astracore") {
    for (auto& s : providers) provider_specs_.push_back({std::move(s), {}});
    report_.backend = "onnxruntime";
  }

  explicit OrtBackend(std::vector<ProviderSpec> providers)
      : env_(ORT_LOGGING_LEVEL_WARNING, "astracore"),
        provider_specs_(std::move(providers)) {
    report_.backend = "onnxruntime";
  }

  const std::string& name() const override {
    static const std::string n = "onnxruntime";
    return n;
  }

  std::unique_ptr<Program> compile(const std::string& onnx_path) override {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(0);     // ORT default.

    // Provider selection — short names mapped to full ORT EP names.
    // Quietly drop any EP that's not in this ORT build.
    auto available = Ort::GetAvailableProviders();
    auto is_available = [&](const std::string& full) {
      for (auto& p : available) if (p == full) return true;
      return false;
    };
    auto expand_alias = [](const std::string& s) -> std::string {
      if (s == "cpu")      return "CPUExecutionProvider";
      if (s == "cuda")     return "CUDAExecutionProvider";
      if (s == "tensorrt" || s == "trt") return "TensorrtExecutionProvider";
      if (s == "openvino" || s == "ov")  return "OpenVINOExecutionProvider";
      if (s == "qnn")      return "QNNExecutionProvider";
      if (s == "coreml")   return "CoreMLExecutionProvider";
      if (s == "dml" || s == "directml") return "DmlExecutionProvider";
      if (s == "rocm")     return "ROCMExecutionProvider";
      return s;             // pass through full names verbatim
    };
    // v0.2 — append per-EP. CUDA / TensorRT / OpenVINO / QNN have
    // typed AppendExecutionProvider_* methods; every other EP can be
    // appended by string name with an options map.
    for (auto& spec : provider_specs_) {
      auto full = expand_alias(spec.name);
      if (full == "CPUExecutionProvider") continue;
      if (!is_available(full)) continue;
      try {
        if (full == "CUDAExecutionProvider") {
          OrtCUDAProviderOptions cuda_opts{};
          for (auto& kv : spec.options) {
            if (kv.first == "device_id") {
              cuda_opts.device_id = std::stoi(kv.second);
            } else if (kv.first == "gpu_mem_limit") {
              cuda_opts.gpu_mem_limit =
                  static_cast<std::size_t>(std::stoull(kv.second));
            } else if (kv.first == "arena_extend_strategy") {
              cuda_opts.arena_extend_strategy =
                  std::stoi(kv.second);
            }
          }
          opts.AppendExecutionProvider_CUDA(cuda_opts);
        } else if (full == "TensorrtExecutionProvider") {
          OrtTensorRTProviderOptions trt_opts{};
          // Most TRT knobs are string-or-bool; construct a temporary
          // map and let ORT parse on our behalf via the V2 API path
          // if needed. v0.2 handles the two most-asked keys inline.
          for (auto& kv : spec.options) {
            if (kv.first == "trt_fp16_enable") {
              trt_opts.trt_fp16_enable =
                  (kv.second == "1" || kv.second == "true");
            } else if (kv.first == "trt_max_workspace_size") {
              trt_opts.trt_max_workspace_size =
                  static_cast<std::size_t>(std::stoull(kv.second));
            }
          }
          opts.AppendExecutionProvider_TensorRT(trt_opts);
        } else {
          // Generic path — string-keyed options dict. Works for
          // OpenVINO / QNN / CoreML / XNNPACK / DML on recent ORT.
          std::unordered_map<std::string, std::string> om(
              spec.options.begin(), spec.options.end());
          opts.AppendExecutionProvider(full, om);
        }
      } catch (const Ort::Exception& exc) {
        // EP present-but-misconfigured → warn + fall through to CPU.
        // (ORT sets a logger; also surface on stderr for visibility.)
        std::fprintf(stderr,
                     "[astracore] EP '%s' rejected options: %s\n",
                     full.c_str(), exc.what());
      }
    }

    auto program = std::make_unique<OrtProgram>();
    program->session = Ort::Session(env_, onnx_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    std::size_t n_in = program->session.GetInputCount();
    std::size_t n_out = program->session.GetOutputCount();
    program->input_names.reserve(n_in);
    program->output_names.reserve(n_out);
    for (std::size_t i = 0; i < n_in; ++i) {
      auto p = program->session.GetInputNameAllocated(i, alloc);
      program->input_names.emplace_back(p.get());
    }
    for (std::size_t i = 0; i < n_out; ++i) {
      auto p = program->session.GetOutputNameAllocated(i, alloc);
      program->output_names.emplace_back(p.get());
    }

    // Record active EPs on the report (ORT C++ API: GetCurrentGpuDeviceId
    // is provider-specific; report what the build advertises).
    report_.active_providers.clear();
    for (auto& p : available) report_.active_providers.push_back(p);

    auto fname = std::filesystem::path(onnx_path).filename().string();
    report_.model = fname;
    return program;
  }

  std::map<std::string, Tensor> run(
      Program& program_base,
      const std::map<std::string, Tensor>& inputs) override {
    auto& program = static_cast<OrtProgram&>(program_base);
    Ort::MemoryInfo mem_info("Cpu", OrtArenaAllocator,
                             0, OrtMemTypeDefault);

    // Build ORT input value list in the order session expects.
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> in_names_cstr;
    ort_inputs.reserve(program.input_names.size());
    in_names_cstr.reserve(program.input_names.size());

    for (auto& name : program.input_names) {
      auto it = inputs.find(name);
      if (it == inputs.end()) {
        throw std::runtime_error(
            "missing input tensor: " + name);
      }
      const Tensor& t = it->second;
      Ort::Value v = Ort::Value::CreateTensor(
          mem_info,
          const_cast<void*>(t.data()),
          t.bytes(),
          t.shape().data(),
          t.shape().size(),
          to_ort(t.dtype()));
      ort_inputs.emplace_back(std::move(v));
      in_names_cstr.push_back(name.c_str());
    }

    std::vector<const char*> out_names_cstr;
    out_names_cstr.reserve(program.output_names.size());
    for (auto& s : program.output_names) out_names_cstr.push_back(s.c_str());

    auto t0 = std::chrono::steady_clock::now();
    auto out_values = program.session.Run(
        Ort::RunOptions{nullptr},
        in_names_cstr.data(), ort_inputs.data(), ort_inputs.size(),
        out_names_cstr.data(), out_names_cstr.size());
    auto t1 = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t1 - t0).count();

    std::map<std::string, Tensor> outputs;
    for (std::size_t i = 0; i < out_values.size(); ++i) {
      auto info = out_values[i].GetTensorTypeAndShapeInfo();
      auto shape64 = info.GetShape();
      std::vector<int64_t> shape(shape64.begin(), shape64.end());
      auto dtype = from_ort(info.GetElementType());
      std::size_t n_elem = info.GetElementCount();
      std::size_t bytes = n_elem * element_size(dtype);
      auto buf = std::shared_ptr<std::uint8_t[]>(
          new std::uint8_t[bytes],
          std::default_delete<std::uint8_t[]>());
      std::memcpy(buf.get(),
                  out_values[i].GetTensorMutableData<void>(),
                  bytes);
      outputs.emplace(program.output_names[i],
                      Tensor(dtype, std::move(shape), buf, bytes));
    }

    report_.n_inferences += 1;
    report_.wall_s_total += wall;
    report_.wall_ms_per_inference =
        report_.wall_s_total / report_.n_inferences * 1e3;
    return outputs;
  }

  const Report& report_last() const override { return report_; }

 private:
  Ort::Env env_;
  std::vector<ProviderSpec> provider_specs_;
  Report report_;
};

// ---------------------------------------------------------------------------
// Factory + version
// ---------------------------------------------------------------------------

std::unique_ptr<Backend> make_backend(
    const std::string& name,
    const std::vector<std::string>& providers) {
  if (name == "onnxruntime" || name == "ort") {
    return std::make_unique<OrtBackend>(providers);
  }
  throw std::invalid_argument(
      "no C++ backend registered for name: " + name);
}

std::unique_ptr<Backend> make_backend_with_options(
    const std::string& name,
    const std::vector<ProviderSpec>& providers) {
  if (name == "onnxruntime" || name == "ort") {
    return std::make_unique<OrtBackend>(providers);
  }
  throw std::invalid_argument(
      "no C++ backend registered for name: " + name);
}

const char* version() { return "0.2.0"; }

}  // namespace astracore
