// #include "../libpouq/index/ivf-sq8.hpp"
#include "../libpouq/index/ivf.hpp"
#include "../libpouq/quantizer.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define BIND_Quantizer(name, type)                                                                                     \
  py::class_<type>(m, name)                                                                                            \
      .def(py::init<size_t, size_t, size_t>(), "Constructor", py::arg("c_bit"), py::arg("q_bit"), py::arg("sub"))      \
      .def(                                                                                                            \
          "train",                                                                                                     \
          [](type &self, const py::array_t<float> &data) { self.train(data.data(), data.size()); },                    \
          py::arg("data"))                                                                                             \
      .def("__getitem__", &type::operator[], py::arg("index"))                                                         \
      .def(                                                                                                            \
          "compute_codes", [](type &self, const py::array_t<float> &data) { return 0; }, py::arg("data"))              \
      .def(                                                                                                            \
          "decode",                                                                                                    \
          [](type &self, int _) {                                                                                      \
            std::vector<float> result(self.size());                                                                    \
            for (size_t i = 0; i < self.size(); i++) {                                                                 \
              result[i] = self[i];                                                                                     \
            }                                                                                                          \
            return result;                                                                                             \
          },                                                                                                           \
          py::arg("data"))                                                                                             \
      .def("size", &type::size)                                                                                        \
      .def(                                                                                                            \
          "l2_distance",                                                                                               \
          [](type &self, const py::array_t<float> &data, size_t idx) {                                                 \
            float  dis    = 0.0f;                                                                                      \
            size_t offset = idx * data.size();                                                                         \
            for (size_t i = 0; i < data.size(); i++) {                                                                 \
              float dif = data.data()[i] - self[offset + i];                                                           \
              dis += dif * dif;                                                                                        \
            }                                                                                                          \
            return dis;                                                                                                \
          },                                                                                                           \
          py::arg("data"),                                                                                             \
          py::arg("i"))

PYBIND11_MODULE(pouq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  using SQQuantizer         = pouq::QuantizerImpl<pouq::Clusterer, pouq::MinMaxOptimizer>;
  using OSQQuantizer        = pouq::QuantizerImpl<pouq::Clusterer, pouq::SGDOptimizer>;
  using OSQ2Quantizer       = pouq::QuantizerImpl<pouq::Clusterer, pouq::PSOptimizer>;
  using POUQKRangeQuantizer = pouq::QuantizerImpl<pouq::KrangeClusterer, pouq::MinMaxOptimizer>;
  using POUQKMeansQuantizer = pouq::QuantizerImpl<pouq::CKmeansClusterer, pouq::MinMaxOptimizer>;
  BIND_Quantizer("SQQuantizer", SQQuantizer);
  BIND_Quantizer("OSQQuantizer", OSQQuantizer);
  BIND_Quantizer("OSQ2Quantizer", OSQ2Quantizer);
  BIND_Quantizer("POUQKRangeQuantizer", POUQKRangeQuantizer);
  BIND_Quantizer("POUQKMeansQuantizer", POUQKMeansQuantizer);
  BIND_Quantizer("POUQQuantizer", pouq::POUQQuantizer);

#define BIND_Index(name, type)                                                                                         \
  py::class_<type>(m, name)                                                                                            \
      .def(py::init<size_t, size_t>(), py::arg("nlist"), py::arg("dim"))                                               \
      .def(                                                                                                            \
          "train",                                                                                                     \
          [](type &self, const py::array_t<float> &data) { self.train(data.data(), data.size()); },                    \
          py::arg("data"))                                                                                             \
      .def(                                                                                                            \
          "search",                                                                                                    \
          [](type &self, const py::array_t<float> &query, size_t k, size_t nprobe) {                                   \
            return self.search(query.data(), k, nprobe);                                                               \
          },                                                                                                           \
          py::arg("query"),                                                                                            \
          py::arg("k"),                                                                                                \
          py::arg("nprobe"));

  BIND_Index("IVFSQ4", IVFSQ4);
  BIND_Index("IVFSQ8", IVFSQ8);

  BIND_Index("IVFPOUQ4", IVFPOUQ4);
  BIND_Index("IVFPOUQ8", IVFPOUQ8);

  m.def("compute_mse", [](const SQQuantizer &sq, const py::array_t<float> &data) {
    return l2distance(sq, data.data(), data.size()) / static_cast<float>(data.size());
  });

  m.def("compute_mse", [](const pouq::POUQQuantizer &sq, const py::array_t<float> &data) {
    return l2distance(sq, data.data(), data.size()) / static_cast<float>(data.size());
  });
}