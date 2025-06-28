#include "../libpouq/quantizer.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class Quantizer final : public pouq::POUQuantizer<> {
public:
  explicit Quantizer(const py::array_t<float> &array,
      const uint64_t                           c_bit,
      const uint64_t                           q_bit,
      const uint64_t                           groups    = 1,
      const bool                               opt_bound = true)
      : pouq::POUQuantizer<>(array.data(), array.size(), c_bit, q_bit, groups, opt_bound) {}
};

float compute_mse(const py::array_t<float> &data, const Quantizer &quantizer) {
  float mse = 0;
  for (uint64_t i = 0; i < static_cast<uint64_t>(data.size()); ++i) {
    const float diff = data.data()[i] - quantizer[i];
    mse += diff * diff;
  }
  return mse / static_cast<float>(data.size());
}

PYBIND11_MODULE(pouq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  py::class_<Quantizer>(m, "Quantizer")
      .def(py::init<const py::array_t<float> &, const uint64_t, const uint64_t, const uint64_t, const bool>(),
          py::arg("data"),
          py::arg("c_bit"),
          py::arg("q_bit"),
          py::arg("groups")    = 1,
          py::arg("opt_bound") = true)
      .def("__getitem__", &Quantizer::operator[], py::arg("i"))
      .def("size", &Quantizer::size);

  m.def("compute_mse", &compute_mse);
}