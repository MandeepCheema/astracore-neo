// SPDX-License-Identifier: Apache-2.0
//
// pybind11 binding — exposes astracore::Backend / Tensor / make_backend
// to Python as the ``astracore_runtime`` extension module.
//
// Once built, Python imports it as:
//
//     from astracore_runtime import make_backend, version
//     be = make_backend("onnxruntime", ["cpu"])
//     program = be.compile("data/models/yolov8n.onnx")
//     out = be.run(program, {"images": numpy_array})
//
// This is the same API surface the Python ``astracore.backends.OrtBackend``
// exposes today — the C++ extension is just a faster, deployment-grade
// implementation of the same contract.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "astracore/runtime.hpp"

#include <cstring>

namespace py = pybind11;
namespace ac = astracore;

namespace {

// ---- numpy <-> astracore::Tensor bridge --------------------------------

ac::DType dtype_from_numpy(py::dtype d) {
  if (d.is(py::dtype::of<float>()))         return ac::DType::kFloat32;
  if (d.is(py::dtype::of<std::int8_t>()))   return ac::DType::kInt8;
  if (d.is(py::dtype::of<std::uint8_t>()))  return ac::DType::kUInt8;
  if (d.is(py::dtype::of<std::int16_t>()))  return ac::DType::kInt16;
  if (d.is(py::dtype::of<std::int32_t>()))  return ac::DType::kInt32;
  if (d.is(py::dtype::of<std::int64_t>()))  return ac::DType::kInt64;
  if (d.is(py::dtype::of<bool>()))          return ac::DType::kBool;
  if (d.is(py::dtype::of<double>()))        return ac::DType::kFloat64;
  // Float16 — numpy ships as 'e' kind; we route via raw bytes.
  if (d.kind() == 'f' && d.itemsize() == 2) return ac::DType::kFloat16;
  throw std::invalid_argument(
      "unsupported numpy dtype: " + std::string(py::str(d)));
}

py::dtype numpy_from_dtype(ac::DType d) {
  switch (d) {
    case ac::DType::kFloat32: return py::dtype::of<float>();
    case ac::DType::kInt8:    return py::dtype::of<std::int8_t>();
    case ac::DType::kUInt8:   return py::dtype::of<std::uint8_t>();
    case ac::DType::kInt16:   return py::dtype::of<std::int16_t>();
    case ac::DType::kInt32:   return py::dtype::of<std::int32_t>();
    case ac::DType::kInt64:   return py::dtype::of<std::int64_t>();
    case ac::DType::kBool:    return py::dtype::of<bool>();
    case ac::DType::kFloat64: return py::dtype::of<double>();
    case ac::DType::kFloat16: return py::dtype("float16");
  }
  return py::dtype::of<float>();
}

// Build an astracore::Tensor view over a numpy array.  The Python
// caller must keep the array alive for the duration of run() — we hold
// a py::object reference until the call returns.
ac::Tensor tensor_from_numpy(py::array arr) {
  auto info = arr.request(false /* writable */);
  std::vector<std::int64_t> shape(info.shape.begin(), info.shape.end());
  return ac::Tensor(
      dtype_from_numpy(arr.dtype()),
      std::move(shape),
      info.ptr,
      static_cast<std::size_t>(info.size * info.itemsize));
}

py::array numpy_from_tensor(const ac::Tensor& t) {
  auto dt = numpy_from_dtype(t.dtype());
  py::array out(dt, t.shape());
  std::memcpy(out.mutable_data(), t.data(), t.bytes());
  return out;
}

}  // namespace

PYBIND11_MODULE(astracore_runtime, m) {
  m.doc() =
      "AstraCore Neo C++ runtime — pybind11 binding to astracore::Backend.";
  m.attr("__version__") = ac::version();
  m.def("version", &ac::version);

  py::enum_<ac::DType>(m, "DType")
      .value("Float32", ac::DType::kFloat32)
      .value("UInt8",   ac::DType::kUInt8)
      .value("Int8",    ac::DType::kInt8)
      .value("Int16",   ac::DType::kInt16)
      .value("Int32",   ac::DType::kInt32)
      .value("Int64",   ac::DType::kInt64)
      .value("Bool",    ac::DType::kBool)
      .value("Float16", ac::DType::kFloat16)
      .value("Float64", ac::DType::kFloat64);

  py::class_<ac::Report>(m, "Report")
      .def_readonly("backend",          &ac::Report::backend)
      .def_readonly("model",            &ac::Report::model)
      .def_readonly("active_providers", &ac::Report::active_providers)
      .def_readonly("n_inferences",     &ac::Report::n_inferences)
      .def_readonly("wall_s_total",     &ac::Report::wall_s_total)
      .def_readonly("wall_ms_per_inference",
                    &ac::Report::wall_ms_per_inference);

  py::class_<ac::Program>(m, "Program");

  py::class_<ac::Backend>(m, "Backend")
      .def("name", [](const ac::Backend& b) { return b.name(); })
      .def("compile", [](ac::Backend& b, const std::string& path) {
            return b.compile(path).release();
          },
          py::return_value_policy::take_ownership)
      .def("run",
           [](ac::Backend& b, ac::Program& prog,
              std::map<std::string, py::array> inputs) {
             // Build C++ tensor views; keep the numpy arrays alive
             // until run() returns.
             std::map<std::string, ac::Tensor> in_tensors;
             for (auto& kv : inputs) {
               // Ensure C-contiguous so info.ptr points at row-major data.
               if (!(kv.second.flags() & py::array::c_style)) {
                 kv.second = py::array(kv.second.dtype(),
                                       kv.second.request().shape,
                                       kv.second.data());
               }
               in_tensors.emplace(kv.first,
                                  tensor_from_numpy(kv.second));
             }
             auto outs = b.run(prog, in_tensors);
             std::map<std::string, py::array> py_out;
             for (auto& kv : outs) {
               py_out.emplace(kv.first, numpy_from_tensor(kv.second));
             }
             return py_out;
           })
      .def("report_last", &ac::Backend::report_last,
           py::return_value_policy::reference_internal);

  m.def("make_backend",
        [](const std::string& name,
           std::vector<std::string> providers) {
          return ac::make_backend(name, std::move(providers)).release();
        },
        py::arg("name") = "onnxruntime",
        py::arg("providers") = std::vector<std::string>{},
        py::return_value_policy::take_ownership);

  // v0.2 — per-EP options. Python signature:
  //   make_backend_with_options("onnxruntime",
  //       [("cuda", {"device_id": "0", "gpu_mem_limit": "2147483648"}),
  //        ("cpu", {})])
  m.def("make_backend_with_options",
        [](const std::string& name,
           std::vector<std::pair<std::string,
                                 std::map<std::string, std::string>>>
               raw_specs) {
          std::vector<ac::ProviderSpec> specs;
          specs.reserve(raw_specs.size());
          for (auto& rp : raw_specs) {
            specs.emplace_back(std::move(rp.first), std::move(rp.second));
          }
          return ac::make_backend_with_options(name, specs).release();
        },
        py::arg("name") = "onnxruntime",
        py::arg("providers") =
            std::vector<std::pair<std::string,
                                  std::map<std::string, std::string>>>{},
        py::return_value_policy::take_ownership);
}
