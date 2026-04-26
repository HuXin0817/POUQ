#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>

#include "../src/quantizer.h"

namespace py = pybind11;

class PyQuantizerWrapper : public pouq::Quantizer {
 public:
  using pouq::Quantizer::Quantizer;

  void Decode(uint32_t n, py::array_t<float>& arr) { pouq::Quantizer::Decode(n, arr.mutable_data()); }
  void Distance(uint32_t n, const py::array_t<float>& arr) { pouq::Quantizer::Distance(n, arr.data()); }
};

PYBIND11_MODULE(pypouq, m) {
  m.doc() = "POUQ quantization library";

  py::class_<PyQuantizerWrapper>(m, "Quantizer")
      .def(py::init<>())
      .def("train", &PyQuantizerWrapper::Train)
      .def("decode", py::overload_cast<uint32_t, py::array_t<float>&>(&PyQuantizerWrapper::Decode))
      .def("distance", py::overload_cast<uint32_t, const py::array_t<float>&>(&PyQuantizerWrapper::Distance))
      .def("clear", &PyQuantizerWrapper::Clear, "Clear the quantizer");
}
