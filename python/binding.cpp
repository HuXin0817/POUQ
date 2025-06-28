#include "../libplsq/ivf.hpp"
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

PYBIND11_MODULE(plsq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  using SQQuantizer         = plsq::QuantizerImpl<plsq::Clusterer, plsq::MinMaxOptimizer>;
  using LSQQuantizer        = plsq::QuantizerImpl<plsq::Clusterer, plsq::SGDOptimizer>;
  using LSQ2Quantizer       = plsq::QuantizerImpl<plsq::Clusterer, plsq::PSOptimizer>;
  using PLSQKRangeQuantizer = plsq::QuantizerImpl<plsq::KrangeClusterer, plsq::MinMaxOptimizer>;
  using PLSQKMeansQuantizer = plsq::QuantizerImpl<plsq::CKmeansClusterer, plsq::MinMaxOptimizer>;
  BIND_Quantizer("SQQuantizer", SQQuantizer);
  BIND_Quantizer("LSQQuantizer", LSQQuantizer);
  BIND_Quantizer("LSQ2Quantizer", LSQ2Quantizer);
  BIND_Quantizer("PLSQKRangeQuantizer", PLSQKRangeQuantizer);
  BIND_Quantizer("PLSQKMeansQuantizer", PLSQKMeansQuantizer);
  BIND_Quantizer("PLSQQuantizer", plsq::PLSQQuantizer);

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

  m.def("compute_mse", [](const plsq::PLSQQuantizer &sq, const py::array_t<float> &data) {
    return l2distance(sq, data.data(), data.size()) / static_cast<float>(data.size());
  });
}