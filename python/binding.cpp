#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/quantizer.h"

namespace py = pybind11;

PYBIND11_MODULE(pypouq, m) {
  m.doc() = "POUQ quantization library";

  using Quantizer = pouq::Quantizer;

  py::class_<Quantizer>(m, "Quantizer")
      .def(py::init<>())
      .def("train", &Quantizer::Train, "Train the quantizer with data")
      .def("decode", py::overload_cast<uint32_t>(&Quantizer::Decode), "Decode data and return as a new vector")
      .def("distance",
           py::overload_cast<uint32_t, const std::vector<float>&>(&Quantizer::Distance),
           "Calculate distance to data")
      .def("clear", &Quantizer::Clear, "Clear the quantizer");
}
