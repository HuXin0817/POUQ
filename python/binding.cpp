#include "../libpouq/quantizer.h"
#include "../libpouq/utils.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

#define GENERATE_POUQ_CONSTRUCTORS(Q)                                                                                  \
  case Q:                                                                                                              \
    switch (c_bit) {                                                                                                   \
      case 0:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<0, Q>>(groups);                                               \
        break;                                                                                                         \
      case 1:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<1, Q>>(groups);                                               \
        break;                                                                                                         \
      case 2:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<2, Q>>(groups);                                               \
        break;                                                                                                         \
      case 3:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<3, Q>>(groups);                                               \
        break;                                                                                                         \
      case 4:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<4, Q>>(groups);                                               \
        break;                                                                                                         \
      case 5:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<5, Q>>(groups);                                               \
        break;                                                                                                         \
      case 6:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<6, Q>>(groups);                                               \
        break;                                                                                                         \
      case 7:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<7, Q>>(groups);                                               \
        break;                                                                                                         \
      case 8:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<8, Q>>(groups);                                               \
        break;                                                                                                         \
      case 9:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<9, Q>>(groups);                                               \
        break;                                                                                                         \
      case 10:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<10, Q>>(groups);                                              \
        break;                                                                                                         \
      case 11:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<11, Q>>(groups);                                              \
        break;                                                                                                         \
      case 12:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<12, Q>>(groups);                                              \
        break;                                                                                                         \
      case 13:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<13, Q>>(groups);                                              \
        break;                                                                                                         \
      case 14:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<14, Q>>(groups);                                              \
        break;                                                                                                         \
      case 15:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<15, Q>>(groups);                                              \
        break;                                                                                                         \
      case 16:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<16, Q>>(groups);                                              \
        break;                                                                                                         \
      default:                                                                                                         \
        throw std::invalid_argument("c_bit must be 0-16");                                                             \
    }                                                                                                                  \
    break;

#define GENERATE_SQ_CONSTRUCTORS(Q)                                                                                    \
  case Q:                                                                                                              \
    quantizer_ = std::make_unique<pouq::ScaledQuantizer<Q>>(groups);                                                   \
    break;

#define GENERATE_ALL_POUQ_CONSTRUCTORS()                                                                               \
  GENERATE_POUQ_CONSTRUCTORS(1)                                                                                        \
  GENERATE_POUQ_CONSTRUCTORS(2)                                                                                        \
  GENERATE_POUQ_CONSTRUCTORS(3)                                                                                        \
  GENERATE_POUQ_CONSTRUCTORS(4)                                                                                        \
  GENERATE_POUQ_CONSTRUCTORS(5)                                                                                        \
  GENERATE_POUQ_CONSTRUCTORS(6)                                                                                        \
  GENERATE_POUQ_CONSTRUCTORS(7)                                                                                        \
  GENERATE_POUQ_CONSTRUCTORS(8)                                                                                        \
  GENERATE_POUQ_CONSTRUCTORS(9)                                                                                        \
  GENERATE_POUQ_CONSTRUCTORS(10)                                                                                       \
  GENERATE_POUQ_CONSTRUCTORS(11)                                                                                       \
  GENERATE_POUQ_CONSTRUCTORS(12)                                                                                       \
  GENERATE_POUQ_CONSTRUCTORS(13)                                                                                       \
  GENERATE_POUQ_CONSTRUCTORS(14)                                                                                       \
  GENERATE_POUQ_CONSTRUCTORS(15)                                                                                       \
  GENERATE_POUQ_CONSTRUCTORS(16)

#define GENERATE_ALL_SQ_CONSTRUCTORS()                                                                                 \
  GENERATE_SQ_CONSTRUCTORS(1)                                                                                          \
  GENERATE_SQ_CONSTRUCTORS(2)                                                                                          \
  GENERATE_SQ_CONSTRUCTORS(3)                                                                                          \
  GENERATE_SQ_CONSTRUCTORS(4)                                                                                          \
  GENERATE_SQ_CONSTRUCTORS(5)                                                                                          \
  GENERATE_SQ_CONSTRUCTORS(6)                                                                                          \
  GENERATE_SQ_CONSTRUCTORS(7)                                                                                          \
  GENERATE_SQ_CONSTRUCTORS(8)                                                                                          \
  GENERATE_SQ_CONSTRUCTORS(9)                                                                                          \
  GENERATE_SQ_CONSTRUCTORS(10)                                                                                         \
  GENERATE_SQ_CONSTRUCTORS(11)                                                                                         \
  GENERATE_SQ_CONSTRUCTORS(12)                                                                                         \
  GENERATE_SQ_CONSTRUCTORS(13)                                                                                         \
  GENERATE_SQ_CONSTRUCTORS(14)                                                                                         \
  GENERATE_SQ_CONSTRUCTORS(15)                                                                                         \
  GENERATE_SQ_CONSTRUCTORS(16)

class POUQuantizer final {
private:
  std::unique_ptr<pouq::Quantizer> quantizer_;

public:
  explicit POUQuantizer(const size_t c_bit, const size_t q_bit, const size_t groups = 1) {
    if (c_bit > 16 || q_bit == 0 || q_bit > 16) {
      throw std::invalid_argument("c_bit must be 0-16, q_bit must be 1-16");
    }

    switch (q_bit) {
      GENERATE_ALL_POUQ_CONSTRUCTORS()
      default:
        throw std::invalid_argument("q_bit must be 1-16");
    }
  }

  void train(const py::array_t<float> &data) { quantizer_->train(data.data(), data.size()); }

  float operator[](size_t i) const { return (*quantizer_)[i]; }

  size_t size() const { return quantizer_->size(); }
};

class ScaledQuantizer final {
private:
  std::unique_ptr<pouq::Quantizer> quantizer_;

public:
  explicit ScaledQuantizer(const size_t q_bit, const size_t groups = 1) {
    if (q_bit == 0 || q_bit > 16) {
      throw std::invalid_argument("q_bit must be 1-16");
    }

    switch (q_bit) {
      GENERATE_ALL_SQ_CONSTRUCTORS()
      default:
        throw std::invalid_argument("q_bit must be 1-32");
    }
  }

  void train(const py::array_t<float> &data) { quantizer_->train(data.data(), data.size()); }

  float operator[](size_t i) const { return (*quantizer_)[i]; }

  size_t size() const { return quantizer_->size(); }
};

float compute_mse(const py::array_t<float> &data, const POUQuantizer &quantizer) {
  return compute_mse(data.data(), quantizer, data.size());
}

float compute_mse(const py::array_t<float> &data, const ScaledQuantizer &quantizer) {
  return compute_mse(data.data(), quantizer, data.size());
}

PYBIND11_MODULE(pouq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  py::class_<POUQuantizer>(m, "POUQuantizer")
      .def(py::init<const size_t, const size_t, const size_t>(),
          py::arg("c_bit"),
          py::arg("q_bit"),
          py::arg("groups") = 1)
      .def("train", &POUQuantizer::train, py::arg("data"))
      .def("__getitem__", &POUQuantizer::operator[], py::arg("i"))
      .def("size", &POUQuantizer::size);

  py::class_<ScaledQuantizer>(m, "ScaledQuantizer")
      .def(py::init<const size_t, const size_t>(), py::arg("q_bit"), py::arg("groups") = 1)
      .def("train", &ScaledQuantizer::train, py::arg("data"))
      .def("__getitem__", &ScaledQuantizer::operator[], py::arg("i"))
      .def("size", &ScaledQuantizer::size);

  m.def("compute_mse", static_cast<float (*)(const py::array_t<float> &, const POUQuantizer &)>(&compute_mse));
  m.def("compute_mse", static_cast<float (*)(const py::array_t<float> &, const ScaledQuantizer &)>(&compute_mse));
}

#undef GENERATE_POUQ_CONSTRUCTORS
#undef GENERATE_SQ_CONSTRUCTORS
#undef GENERATE_ALL_POUQ_CONSTRUCTORS
#undef GENERATE_ALL_SQ_CONSTRUCTORS