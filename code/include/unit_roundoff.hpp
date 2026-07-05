#pragma once

#include <limits>
#include <type_traits>

#include "hdnum.hh"

namespace mpir {

constexpr double pow2_neg(int p)
{
    double x = 1.0;
    for (int i = 0; i < p; ++i) {
        x *= 0.5;
    }
    return x;
}

namespace detail {

template<class>
struct unit_roundoff_always_false : std::false_type {};

} // namespace detail

template<class T>
struct unit_roundoff_traits {
    static_assert(detail::unit_roundoff_always_false<T>::value,
        "unit_roundoff_traits: unsupported scalar type");
};

template<>
struct unit_roundoff_traits<hdnum::FP32> {
    static constexpr double value =
        0.5 * static_cast<double>(std::numeric_limits<hdnum::FP32>::epsilon());
};

template<>
struct unit_roundoff_traits<hdnum::FP64> {
    static constexpr double value =
        0.5 * static_cast<double>(std::numeric_limits<hdnum::FP64>::epsilon());
};

#ifdef HDNUM_HAS_CPFLOAT
template<int m, int e>
struct unit_roundoff_traits<hdnum::CPFloat<m,e>> {
    static constexpr double value = pow2_neg(m);
};
#endif

#ifdef HDNUM_HAS_GMP
template<int m>
struct unit_roundoff_traits<hdnum::FP<m>> {
    static constexpr double value = pow2_neg(m);
};
#endif

template<class T>
constexpr double default_unit_roundoff()
{
    return unit_roundoff_traits<T>::value;
}

} // namespace mpir