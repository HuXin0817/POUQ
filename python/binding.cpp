#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/quantizer.h"

namespace py = pybind11;

PYBIND11_MODULE(pypouq, m) {
  m.doc() = "POUQ quantization library";

  using Quantizer = pouq::Quantizer;

  py::class_<Quantizer>(m, "Quantizer")
      .def(py::init<>())
      .def("train", &Quantizer::Train)
      .def("decode", py::overload_cast<uint32_t>(&Quantizer::Decode))
      .def("distance", py::overload_cast<uint32_t, const std::vector<float>&>(&Quantizer::Distance))
      .def("clear", &Quantizer::Clear, "Clear the quantizer");
}
