#include "../libpouq/quantizer.hpp"
#include "../libpouq/utils.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define BIND_QUANTIZER_METHODS(Q)                                                                                      \
  .def(                                                                                                                \
      "train",                                                                                                         \
      [](Q &self, const py::array_t<float> &array) { self.train(array.data(), array.size()); },                        \
      py::arg("data"))                                                                                                 \
      .def("__getitem__", &Q::operator[], py::arg("i"))                                                                \
      .def("size", &Q::size)

#define BIND_COMPUTE_MSE(Q)                                                                                            \
  m.def("compute_mse", [](const py::array_t<float> &array, const Q &quantizer) {                                       \
    return compute_mse(array.data(), quantizer, array.size());                                                         \
  });

PYBIND11_MODULE(pouq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  py::class_<pouq::Float32Quantizer>(m, "Float32Quantizer") BIND_QUANTIZER_METHODS(pouq::Float32Quantizer);

  py::class_<pouq::SQQuantizer>(m, "SQQuantizer")
      .def(py::init<size_t, size_t>(), py::arg("q_bit"), py::arg("groups") = 1)
          BIND_QUANTIZER_METHODS(pouq::SQQuantizer);

  py::class_<pouq::OSQQuantizer>(m, "OSQQuantizer")
      .def(py::init<size_t, size_t>(), py::arg("q_bit"), py::arg("groups") = 1)
          BIND_QUANTIZER_METHODS(pouq::OSQQuantizer);

  py::class_<pouq::LloydMaxQuantizer>(m, "LloydMaxQuantizer")
      .def(py::init<size_t, size_t>(), py::arg("c_bit"), py::arg("groups") = 1)
          BIND_QUANTIZER_METHODS(pouq::LloydMaxQuantizer);

  py::class_<pouq::POUQQuantizer>(m, "POUQQuantizer")
      .def(py::init<size_t, size_t, size_t>(), py::arg("c_bit"), py::arg("q_bit"), py::arg("groups") = 1)
          BIND_QUANTIZER_METHODS(pouq::POUQQuantizer);

  BIND_COMPUTE_MSE(pouq::Float32Quantizer)
  BIND_COMPUTE_MSE(pouq::SQQuantizer)
  BIND_COMPUTE_MSE(pouq::OSQQuantizer)
  BIND_COMPUTE_MSE(pouq::LloydMaxQuantizer)
  BIND_COMPUTE_MSE(pouq::POUQQuantizer)
}