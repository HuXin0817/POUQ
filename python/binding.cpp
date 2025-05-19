#include "../libpouq/qvector.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pouq, m) {
  m.doc() = "Python bindings for the POUQ class";

  py::class_<QVector>(m, "QVector")
      .def(py::init<const py::array_t<float> &,
               const uint64_t,
               const uint64_t,
               const bool,
               const uint64_t,
               const uint64_t,
               const float,
               const float,
               const float,
               const float,
               const float>(),
          py::arg("data"),
          py::arg("c_bit"),
          py::arg("q_bit"),
          py::arg("optimize_bound")    = true,
          py::arg("max_iter")          = 128,
          py::arg("grid_side_length")  = 8,
          py::arg("grid_scale_factor") = 0.1f,
          py::arg("initial_inertia")   = 0.9f,
          py::arg("final_inertia")     = 0.4f,
          py::arg("c1")                = 1.8f,
          py::arg("c2")                = 1.8f,
          "Initialize QVector.")
      .def("at", &QVector::at, py::arg("i"), "Get value at index i")
      .def("ndim", &QVector::ndim, "Get ndim of the vector")
      .def("shape", &QVector::shape, "Get shape of the vector");
}