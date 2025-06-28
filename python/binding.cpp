#include "../libpouq/quantizer.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

class QuantizerWrapper {
public:
  virtual ~QuantizerWrapper()               = default;
  virtual float  operator[](size_t i) const = 0;
  virtual size_t size() const               = 0;
};

template <size_t c_bit, size_t q_bit>
class QuantizerImpl final : public QuantizerWrapper {
private:
  std::unique_ptr<pouq::POUQuantizer<c_bit, q_bit>> quantizer_;

public:
  explicit QuantizerImpl(const py::array_t<float> &array, const size_t groups = 1, const bool opt_bound = true)
      : quantizer_(std::make_unique<pouq::POUQuantizer<c_bit, q_bit>>(array.data(), array.size(), groups, opt_bound)) {}

  float operator[](size_t i) const override { return (*quantizer_)[i]; }

  size_t size() const override { return quantizer_->size(); }
};

#define GENERATE_C_COMBINATIONS(Q)                                                                                     \
  case Q:                                                                                                              \
    switch (c_bit) {                                                                                                   \
      case 0:                                                                                                          \
        return std::make_unique<QuantizerImpl<0, Q>>(array, groups, opt_bound);                                        \
      case 1:                                                                                                          \
        return std::make_unique<QuantizerImpl<1, Q>>(array, groups, opt_bound);                                        \
      case 2:                                                                                                          \
        return std::make_unique<QuantizerImpl<2, Q>>(array, groups, opt_bound);                                        \
      case 3:                                                                                                          \
        return std::make_unique<QuantizerImpl<3, Q>>(array, groups, opt_bound);                                        \
      case 4:                                                                                                          \
        return std::make_unique<QuantizerImpl<4, Q>>(array, groups, opt_bound);                                        \
      case 5:                                                                                                          \
        return std::make_unique<QuantizerImpl<5, Q>>(array, groups, opt_bound);                                        \
      case 6:                                                                                                          \
        return std::make_unique<QuantizerImpl<6, Q>>(array, groups, opt_bound);                                        \
      case 7:                                                                                                          \
        return std::make_unique<QuantizerImpl<7, Q>>(array, groups, opt_bound);                                        \
      case 8:                                                                                                          \
        return std::make_unique<QuantizerImpl<8, Q>>(array, groups, opt_bound);                                        \
      case 9:                                                                                                          \
        return std::make_unique<QuantizerImpl<9, Q>>(array, groups, opt_bound);                                        \
      case 10:                                                                                                         \
        return std::make_unique<QuantizerImpl<10, Q>>(array, groups, opt_bound);                                       \
      case 11:                                                                                                         \
        return std::make_unique<QuantizerImpl<11, Q>>(array, groups, opt_bound);                                       \
      case 12:                                                                                                         \
        return std::make_unique<QuantizerImpl<12, Q>>(array, groups, opt_bound);                                       \
      case 13:                                                                                                         \
        return std::make_unique<QuantizerImpl<13, Q>>(array, groups, opt_bound);                                       \
      case 14:                                                                                                         \
        return std::make_unique<QuantizerImpl<14, Q>>(array, groups, opt_bound);                                       \
      case 15:                                                                                                         \
        return std::make_unique<QuantizerImpl<15, Q>>(array, groups, opt_bound);                                       \
      case 16:                                                                                                         \
        return std::make_unique<QuantizerImpl<16, Q>>(array, groups, opt_bound);                                       \
      case 17:                                                                                                         \
        return std::make_unique<QuantizerImpl<17, Q>>(array, groups, opt_bound);                                       \
      case 18:                                                                                                         \
        return std::make_unique<QuantizerImpl<18, Q>>(array, groups, opt_bound);                                       \
      case 19:                                                                                                         \
        return std::make_unique<QuantizerImpl<19, Q>>(array, groups, opt_bound);                                       \
      case 20:                                                                                                         \
        return std::make_unique<QuantizerImpl<20, Q>>(array, groups, opt_bound);                                       \
      case 21:                                                                                                         \
        return std::make_unique<QuantizerImpl<21, Q>>(array, groups, opt_bound);                                       \
      case 22:                                                                                                         \
        return std::make_unique<QuantizerImpl<22, Q>>(array, groups, opt_bound);                                       \
      case 23:                                                                                                         \
        return std::make_unique<QuantizerImpl<23, Q>>(array, groups, opt_bound);                                       \
      case 24:                                                                                                         \
        return std::make_unique<QuantizerImpl<24, Q>>(array, groups, opt_bound);                                       \
      case 25:                                                                                                         \
        return std::make_unique<QuantizerImpl<25, Q>>(array, groups, opt_bound);                                       \
      case 26:                                                                                                         \
        return std::make_unique<QuantizerImpl<26, Q>>(array, groups, opt_bound);                                       \
      case 27:                                                                                                         \
        return std::make_unique<QuantizerImpl<27, Q>>(array, groups, opt_bound);                                       \
      case 28:                                                                                                         \
        return std::make_unique<QuantizerImpl<28, Q>>(array, groups, opt_bound);                                       \
      case 29:                                                                                                         \
        return std::make_unique<QuantizerImpl<29, Q>>(array, groups, opt_bound);                                       \
      case 30:                                                                                                         \
        return std::make_unique<QuantizerImpl<30, Q>>(array, groups, opt_bound);                                       \
      case 31:                                                                                                         \
        return std::make_unique<QuantizerImpl<31, Q>>(array, groups, opt_bound);                                       \
      case 32:                                                                                                         \
        return std::make_unique<QuantizerImpl<32, Q>>(array, groups, opt_bound);                                       \
      default:                                                                                                         \
        throw std::invalid_argument("c_bit must be 0-32");                                                             \
    }                                                                                                                  \
    break;

#define GENERATE_ALL_COMBINATIONS()                                                                                    \
  GENERATE_C_COMBINATIONS(1)                                                                                           \
  GENERATE_C_COMBINATIONS(2)                                                                                           \
  GENERATE_C_COMBINATIONS(3)                                                                                           \
  GENERATE_C_COMBINATIONS(4)                                                                                           \
  GENERATE_C_COMBINATIONS(5)                                                                                           \
  GENERATE_C_COMBINATIONS(6)                                                                                           \
  GENERATE_C_COMBINATIONS(7)                                                                                           \
  GENERATE_C_COMBINATIONS(8)                                                                                           \
  GENERATE_C_COMBINATIONS(9)                                                                                           \
  GENERATE_C_COMBINATIONS(10)                                                                                          \
  GENERATE_C_COMBINATIONS(11)                                                                                          \
  GENERATE_C_COMBINATIONS(12)                                                                                          \
  GENERATE_C_COMBINATIONS(13)                                                                                          \
  GENERATE_C_COMBINATIONS(14)                                                                                          \
  GENERATE_C_COMBINATIONS(15)                                                                                          \
  GENERATE_C_COMBINATIONS(16)                                                                                          \
  GENERATE_C_COMBINATIONS(17)                                                                                          \
  GENERATE_C_COMBINATIONS(18)                                                                                          \
  GENERATE_C_COMBINATIONS(19)                                                                                          \
  GENERATE_C_COMBINATIONS(20)                                                                                          \
  GENERATE_C_COMBINATIONS(21)                                                                                          \
  GENERATE_C_COMBINATIONS(22)                                                                                          \
  GENERATE_C_COMBINATIONS(23)                                                                                          \
  GENERATE_C_COMBINATIONS(24)                                                                                          \
  GENERATE_C_COMBINATIONS(25)                                                                                          \
  GENERATE_C_COMBINATIONS(26)                                                                                          \
  GENERATE_C_COMBINATIONS(27)                                                                                          \
  GENERATE_C_COMBINATIONS(28)                                                                                          \
  GENERATE_C_COMBINATIONS(29)                                                                                          \
  GENERATE_C_COMBINATIONS(30)                                                                                          \
  GENERATE_C_COMBINATIONS(31)                                                                                          \
  GENERATE_C_COMBINATIONS(32)

std::unique_ptr<QuantizerWrapper> create_quantizer(const py::array_t<float> &array,
    const size_t                                                             c_bit,
    const size_t                                                             q_bit,
    const size_t                                                             groups    = 1,
    const bool                                                               opt_bound = true) {

  if (c_bit > 32 || q_bit == 0 || q_bit > 32) {
    throw std::invalid_argument("c_bit must be 0-32, q_bit must be 1-32");
  }

  switch (q_bit) {
    GENERATE_ALL_COMBINATIONS()
    default:
      throw std::invalid_argument("q_bit must be 1-32");
  }
}

float compute_mse(const py::array_t<float> &data, const QuantizerWrapper &quantizer) {
  float mse = 0;
  for (size_t i = 0; i < static_cast<size_t>(data.size()); ++i) {
    const float diff = data.data()[i] - quantizer[i];
    mse += diff * diff;
  }
  return mse / static_cast<float>(data.size());
}

PYBIND11_MODULE(pouq, m) {
  m.doc() = R"pbdoc(Piecewise-Optimized Uniform Quantization (POUQ))pbdoc";

  py::class_<QuantizerWrapper>(m, "Quantizer")
      .def("__getitem__", &QuantizerWrapper::operator[], py::arg("i"))
      .def("size", &QuantizerWrapper::size);

  m.def("create_quantizer",
      &create_quantizer,
      py::arg("data"),
      py::arg("c_bit"),
      py::arg("q_bit"),
      py::arg("groups")    = 1,
      py::arg("opt_bound") = true,
      "Create a quantizer with specified c_bit and q_bit parameters");

  m.def("compute_mse", &compute_mse);
}

#undef GENERATE_C_COMBINATIONS
#undef GENERATE_ALL_COMBINATIONS