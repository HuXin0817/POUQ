#include "../libpouq/quantizer.h"
#include "../libpouq/utils.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define BIND_POU_QUANTIZER_SINGLE(c_bit, q_bit)                                                                        \
  py::class_<pouq::POUQuantizer<c_bit, q_bit>>(m, "POUQuantizer" #c_bit "_" #q_bit)                                    \
      .def(py::init<size_t>(), py::arg("groups") = 1)                                                                  \
      .def(                                                                                                            \
          "train",                                                                                                     \
          [](pouq::POUQuantizer<c_bit, q_bit> &self, const py::array_t<float> &data) {                                 \
            self.train(data.data(), data.size());                                                                      \
          },                                                                                                           \
          py::arg("data"))                                                                                             \
      .def("__getitem__", &pouq::POUQuantizer<c_bit, q_bit>::operator[], py::arg("i"))                                 \
      .def("size", &pouq::POUQuantizer<c_bit, q_bit>::size);                                                           \
  m.def("compute_mse", [](const py::array_t<float> &data, const pouq::POUQuantizer<c_bit, q_bit> &quantizer) {         \
    return compute_mse(data.data(), quantizer, data.size());                                                           \
  });

#define BIND_POU_QUANTIZER_ALL_Q(c_bit)                                                                                \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 1)                                                                                  \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 2)                                                                                  \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 3)                                                                                  \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 4)                                                                                  \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 5)                                                                                  \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 6)                                                                                  \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 7)                                                                                  \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 8)                                                                                  \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 9)                                                                                  \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 10)                                                                                 \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 11)                                                                                 \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 12)                                                                                 \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 13)                                                                                 \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 14)                                                                                 \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 15)                                                                                 \
  BIND_POU_QUANTIZER_SINGLE(c_bit, 16)

#define BIND_SCALED_QUANTIZER(q_bit)                                                                                   \
  py::class_<pouq::ScaledQuantizer<q_bit>>(m, "ScaledQuantizer" #q_bit)                                                \
      .def(py::init<const size_t>(), py::arg("groups") = 1)                                                            \
      .def(                                                                                                            \
          "train",                                                                                                     \
          [](pouq::ScaledQuantizer<q_bit> &self, const py::array_t<float> &data) {                                     \
            self.train(data.data(), data.size());                                                                      \
          },                                                                                                           \
          py::arg("data"))                                                                                             \
      .def("__getitem__", &pouq::ScaledQuantizer<q_bit>::operator[], py::arg("i"))                                     \
      .def("size", &pouq::ScaledQuantizer<q_bit>::size);                                                               \
  m.def("compute_mse", [](const py::array_t<float> &data, const pouq::ScaledQuantizer<q_bit> &quantizer) {             \
    return compute_mse(data.data(), quantizer, data.size());                                                           \
  });

PYBIND11_MODULE(pouq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  BIND_POU_QUANTIZER_ALL_Q(0)
  BIND_POU_QUANTIZER_ALL_Q(1)
  BIND_POU_QUANTIZER_ALL_Q(2)
  BIND_POU_QUANTIZER_ALL_Q(3)
  BIND_POU_QUANTIZER_ALL_Q(4)
  BIND_POU_QUANTIZER_ALL_Q(5)
  BIND_POU_QUANTIZER_ALL_Q(6)
  BIND_POU_QUANTIZER_ALL_Q(7)
  BIND_POU_QUANTIZER_ALL_Q(8)
  BIND_POU_QUANTIZER_ALL_Q(9)
  BIND_POU_QUANTIZER_ALL_Q(10)
  BIND_POU_QUANTIZER_ALL_Q(11)
  BIND_POU_QUANTIZER_ALL_Q(12)
  BIND_POU_QUANTIZER_ALL_Q(13)
  BIND_POU_QUANTIZER_ALL_Q(14)
  BIND_POU_QUANTIZER_ALL_Q(15)
  BIND_POU_QUANTIZER_ALL_Q(16)

  BIND_SCALED_QUANTIZER(1)
  BIND_SCALED_QUANTIZER(2)
  BIND_SCALED_QUANTIZER(3)
  BIND_SCALED_QUANTIZER(4)
  BIND_SCALED_QUANTIZER(5)
  BIND_SCALED_QUANTIZER(6)
  BIND_SCALED_QUANTIZER(7)
  BIND_SCALED_QUANTIZER(8)
  BIND_SCALED_QUANTIZER(9)
  BIND_SCALED_QUANTIZER(10)
  BIND_SCALED_QUANTIZER(11)
  BIND_SCALED_QUANTIZER(12)
  BIND_SCALED_QUANTIZER(13)
  BIND_SCALED_QUANTIZER(14)
  BIND_SCALED_QUANTIZER(15)
  BIND_SCALED_QUANTIZER(16)
}
