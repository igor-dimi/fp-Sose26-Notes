#pragma once

#include <limits>
#include <type_traits>

#include "hdnum.hh"

namespace mpir {

// Compute 2^(-p) using arithmetic in T.
template<class T>
T pow2_neg(int p)
{
    T x(1);
    const T half(0.5);

    for (int i = 0; i < p; ++i) {
        x *= half;
    }

    return x;
}

namespace detail {

template<class>
struct unit_roundoff_always_false : std::false_type {};

} // namespace detail


// Primary template: unsupported type
template<class T>
struct unit_roundoff_traits {
    static T value()
    {
        static_assert(
            detail::unit_roundoff_always_false<T>::value,
            "unit_roundoff_traits: unsupported scalar type"
        );

        return T{}; // Required syntactically; never reached.
    }
};


// FP32
template<>
struct unit_roundoff_traits<hdnum::FP32> {
    static hdnum::FP32 value()
    {
        return std::numeric_limits<hdnum::FP32>::epsilon()
             / hdnum::FP32(2);
    }
};


// FP64
template<>
struct unit_roundoff_traits<hdnum::FP64> {
    static hdnum::FP64 value()
    {
        return std::numeric_limits<hdnum::FP64>::epsilon()
             / hdnum::FP64(2);
    }
};


#ifdef HDNUM_HAS_CPFLOAT

template<int m, int e>
struct unit_roundoff_traits<hdnum::CPFloat<m, e>> {
    using scalar_type = hdnum::CPFloat<m, e>;

    static scalar_type value()
    {
        return pow2_neg<scalar_type>(m);
    }
};

#endif


#ifdef HDNUM_HAS_GMP

template<int m>
struct unit_roundoff_traits<hdnum::FP<m>> {
    using scalar_type = hdnum::FP<m>;

    static scalar_type value()
    {
        return pow2_neg<scalar_type>(m);
    }
};

#endif


template<class T>
T default_unit_roundoff()
{
    return unit_roundoff_traits<T>::value();
}

} // namespace mpir