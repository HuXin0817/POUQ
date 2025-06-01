#include "../libpouq/quantizer.h"
#include "../libpouq/utils.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define BIND_QUANTIZER_METHODS(quantizer_type)                                                                         \
  .def(                                                                                                                \
      "train",                                                                                                         \
      [](quantizer_type &self, const py::array_t<float> &array) { self.train(array.data(), array.size()); },           \
      py::arg("data"))                                                                                                 \
      .def("__getitem__", &quantizer_type::operator[], py::arg("i"))                                                   \
      .def("size", &quantizer_type::size)

#define BIND_COMPUTE_MSE(quantizer_type)                                                                               \
  m.def("compute_mse", [](const py::array_t<float> &array, const quantizer_type &quantizer) {                          \
    return compute_mse(array.data(), quantizer, array.size());                                                         \
  });

PYBIND11_MODULE(pouq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  py::class_<pouq::Float32Quantizer>(m, "Float32Quantizer")
      .def(py::init<>()) BIND_QUANTIZER_METHODS(pouq::Float32Quantizer);

  py::class_<pouq::ScaledQuantizer>(m, "ScaledQuantizer")
      .def(py::init<const size_t, const size_t>(), py::arg("q_bit"), py::arg("groups") = 1)
          BIND_QUANTIZER_METHODS(pouq::ScaledQuantizer);

  py::class_<pouq::POUQuantizer<>>(m, "POUQuantizer")
      .def(py::init<const size_t, const size_t, const size_t>(),
          py::arg("c_bit"),
          py::arg("q_bit"),
          py::arg("groups") = 1) BIND_QUANTIZER_METHODS(pouq::POUQuantizer<>);

  BIND_COMPUTE_MSE(pouq::POUQuantizer<>)
  BIND_COMPUTE_MSE(pouq::ScaledQuantizer)
}