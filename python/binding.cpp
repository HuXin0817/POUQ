#include "../libpouq/quantizer.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

#define GENERATE_C_CONSTRUCTORS(Q)                                                                                     \
  case Q:                                                                                                              \
    switch (c_bit) {                                                                                                   \
      case 0:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<0, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 1:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<1, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 2:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<2, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 3:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<3, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 4:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<4, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 5:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<5, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 6:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<6, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 7:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<7, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 8:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<8, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 9:                                                                                                          \
        quantizer_ = std::make_unique<pouq::POUQuantizer<9, Q>>(array.data(), array.size(), groups, opt_bound);        \
        break;                                                                                                         \
      case 10:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<10, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 11:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<11, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 12:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<12, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 13:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<13, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 14:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<14, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 15:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<15, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 16:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<16, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 17:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<17, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 18:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<18, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 19:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<19, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 20:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<20, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 21:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<21, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 22:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<22, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 23:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<23, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 24:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<24, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 25:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<25, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 26:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<26, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 27:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<27, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 28:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<28, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 29:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<29, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 30:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<30, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 31:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<31, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      case 32:                                                                                                         \
        quantizer_ = std::make_unique<pouq::POUQuantizer<32, Q>>(array.data(), array.size(), groups, opt_bound);       \
        break;                                                                                                         \
      default:                                                                                                         \
        throw std::invalid_argument("c_bit must be 0-32");                                                             \
    }                                                                                                                  \
    break;

#define GENERATE_ALL_CONSTRUCTORS()                                                                                    \
  GENERATE_C_CONSTRUCTORS(1)                                                                                           \
  GENERATE_C_CONSTRUCTORS(2)                                                                                           \
  GENERATE_C_CONSTRUCTORS(3)                                                                                           \
  GENERATE_C_CONSTRUCTORS(4)                                                                                           \
  GENERATE_C_CONSTRUCTORS(5)                                                                                           \
  GENERATE_C_CONSTRUCTORS(6)                                                                                           \
  GENERATE_C_CONSTRUCTORS(7)                                                                                           \
  GENERATE_C_CONSTRUCTORS(8)                                                                                           \
  GENERATE_C_CONSTRUCTORS(9)                                                                                           \
  GENERATE_C_CONSTRUCTORS(10)                                                                                          \
  GENERATE_C_CONSTRUCTORS(11)                                                                                          \
  GENERATE_C_CONSTRUCTORS(12)                                                                                          \
  GENERATE_C_CONSTRUCTORS(13)                                                                                          \
  GENERATE_C_CONSTRUCTORS(14)                                                                                          \
  GENERATE_C_CONSTRUCTORS(15)                                                                                          \
  GENERATE_C_CONSTRUCTORS(16)                                                                                          \
  GENERATE_C_CONSTRUCTORS(17)                                                                                          \
  GENERATE_C_CONSTRUCTORS(18)                                                                                          \
  GENERATE_C_CONSTRUCTORS(19)                                                                                          \
  GENERATE_C_CONSTRUCTORS(20)                                                                                          \
  GENERATE_C_CONSTRUCTORS(21)                                                                                          \
  GENERATE_C_CONSTRUCTORS(22)                                                                                          \
  GENERATE_C_CONSTRUCTORS(23)                                                                                          \
  GENERATE_C_CONSTRUCTORS(24)                                                                                          \
  GENERATE_C_CONSTRUCTORS(25)                                                                                          \
  GENERATE_C_CONSTRUCTORS(26)                                                                                          \
  GENERATE_C_CONSTRUCTORS(27)                                                                                          \
  GENERATE_C_CONSTRUCTORS(28)                                                                                          \
  GENERATE_C_CONSTRUCTORS(29)                                                                                          \
  GENERATE_C_CONSTRUCTORS(30)                                                                                          \
  GENERATE_C_CONSTRUCTORS(31)                                                                                          \
  GENERATE_C_CONSTRUCTORS(32)

class Quantizer final {
private:
  std::unique_ptr<pouq::Quantizer> quantizer_;

public:
  explicit Quantizer(const py::array_t<float> &array,
      const size_t                             c_bit,
      const size_t                             q_bit,
      const size_t                             groups    = 1,
      const bool                               opt_bound = true) {

    if (c_bit > 32 || q_bit == 0 || q_bit > 32) {
      throw std::invalid_argument("c_bit must be 0-32, q_bit must be 1-32");
    }

    switch (q_bit) {
      GENERATE_ALL_CONSTRUCTORS()
      default:
        throw std::invalid_argument("q_bit must be 1-32");
    }
  }

  float operator[](size_t i) const { return (*quantizer_)[i]; }

  size_t size() const { return quantizer_->size(); }
};

float compute_mse(const py::array_t<float> &data, const Quantizer &quantizer) {
  float mse = 0;
  for (size_t i = 0; i < static_cast<size_t>(data.size()); ++i) {
    const float diff = data.data()[i] - quantizer[i];
    mse += diff * diff;
  }
  return mse / static_cast<float>(data.size());
}

PYBIND11_MODULE(pouq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  py::class_<Quantizer>(m, "Quantizer")
      .def(py::init<const py::array_t<float> &, const size_t, const size_t, const size_t, const bool>(),
          py::arg("data"),
          py::arg("c_bit"),
          py::arg("q_bit"),
          py::arg("groups")    = 1,
          py::arg("opt_bound") = true)
      .def("__getitem__", &Quantizer::operator[], py::arg("i"))
      .def("size", &Quantizer::size);

  m.def("compute_mse", &compute_mse);
}

#undef GENERATE_C_CONSTRUCTORS
#undef GENERATE_ALL_CONSTRUCTORS