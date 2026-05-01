#pragma once

#include <type_traits>
#include <xsimd/xsimd.hpp>

namespace pouq {

namespace detail {

template <size_t N>
struct has_alignment {
  template <class Arch>
  using apply = std::bool_constant<Arch::alignment() == N>;
};

template <class ArchList, template <class> class Cond>
struct find_first;

template <template <class> class Cond>
struct find_first<xsimd::arch_list<>, Cond> {
  using type = xsimd::unavailable;
};

template <class Arch, class... Archs, template <class> class Cond>
struct find_first<xsimd::arch_list<Arch, Archs...>, Cond> {
  using type = std::conditional_t<Cond<Arch>::value, Arch, typename find_first<xsimd::arch_list<Archs...>, Cond>::type>;
};

using m128_arch = find_first<xsimd::supported_architectures, has_alignment<16>::apply>::type;

}  // namespace detail

using m128 = xsimd::batch<float, detail::m128_arch>;

static_assert(m128::size == 4, "Expected 4 floats for 128-bit alignment");

}  // namespace pouq