#include "../libposq/ivf.hpp"
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

PYBIND11_MODULE(posq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  using SQQuantizer         = posq::QuantizerImpl<posq::Clusterer, posq::MinMaxOptimizer>;
  using OSQQuantizer        = posq::QuantizerImpl<posq::Clusterer, posq::SGDOptimizer>;
  using OSQ2Quantizer       = posq::QuantizerImpl<posq::Clusterer, posq::PSOptimizer>;
  using POSQKRangeQuantizer = posq::QuantizerImpl<posq::KrangeClusterer, posq::MinMaxOptimizer>;
  using POSQKMeansQuantizer = posq::QuantizerImpl<posq::CKmeansClusterer, posq::MinMaxOptimizer>;
  BIND_Quantizer("SQQuantizer", SQQuantizer);
  BIND_Quantizer("OSQQuantizer", OSQQuantizer);
  BIND_Quantizer("OSQ2Quantizer", OSQ2Quantizer);
  BIND_Quantizer("POSQKRangeQuantizer", POSQKRangeQuantizer);
  BIND_Quantizer("POSQKMeansQuantizer", POSQKMeansQuantizer);
  BIND_Quantizer("POSQQuantizer", posq::POSQQuantizer);

  py::class_<IvfIndex>(m, "IvfIndex")
      .def(py::init<size_t, size_t>(), py::arg("nlist"), py::arg("dim"))
      .def(
          "train",
          [](IvfIndex &self, const py::array_t<float> &data) { self.train(data.data(), data.size()); },
          py::arg("data"))
      .def(
          "search",
          [](IvfIndex &self, const py::array_t<float> &query, size_t k, size_t nprobe) {
            return self.search(query.data(), k, nprobe);
          },
          py::arg("query"),
          py::arg("k"),
          py::arg("nprobe"));

  m.def("compute_mse", [](const SQQuantizer &sq, const py::array_t<float> &data) {
    return l2distance(sq, data.data(), data.size()) / static_cast<float>(data.size());
  });

  m.def("compute_mse", [](const posq::POSQQuantizer &sq, const py::array_t<float> &data) {
    return l2distance(sq, data.data(), data.size()) / static_cast<float>(data.size());
  });
}