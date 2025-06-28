#include "../libpouq/qvector.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

float compute_mse(const py::array_t<float> &data, const pouq::QVector &q_vector) {
  float mse = 0;
  for (uint64_t i = 0; i < static_cast<uint64_t>(data.size()); ++i) {
    const float dif = data.data()[i] - q_vector[i];
    mse += dif * dif;
  }
  return mse / static_cast<float>(data.size());
}

PYBIND11_MODULE(pouq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  using QVector = pouq::QVector;

  py::class_<QVector>(m, "QVector")
      .def(py::init<const py::array_t<float> &, const uint64_t, const uint64_t, const uint64_t, const bool>(),
          py::arg("data"),
          py::arg("c_bit"),
          py::arg("q_bit"),
          py::arg("groups")    = 1,
          py::arg("opt_bound") = true)
      .def("__getitem__", &QVector::operator[], py::arg("i"))
      .def("size", &QVector::size);

  m.def("compute_mse", &compute_mse);
}