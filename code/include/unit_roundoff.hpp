#pragma once

/**
 * @file unit_roundoff.hpp
 * @brief Unit-roundoff values for the numerical scalar types used by MPIR.
 *
 * This header provides a uniform trait-based interface for obtaining the unit
 * roundoff u of an HDNUM scalar type. FP32 and FP64 use half of
 * std::numeric_limits<T>::epsilon(); parameterized CPFloat and GMP-backed types
 * use u = 2^(-m), where m is their precision parameter. Unsupported types are
 * rejected at compile time. These values are used for default refinement
 * tolerances and theoretical condition-number boundaries.
 */

#include <limits>
#include <type_traits>

#include "hdnum.hh"

namespace mpir {


/**
 * @brief Computes 2^(-p) using arithmetic in T.
 *
 * Repeated multiplication by one half avoids requiring a compatible std::pow
 * overload for custom scalar types.
 *
 * @tparam T Arithmetic and result type.
 * @param p Nonnegative exponent magnitude.
 * @return 2^(-p) represented in T.
 *
 * @pre p >= 0.
 */
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

/**
 * @brief Dependent false value used to reject unsupported scalar types.
 *
 * The unnamed template parameter delays evaluation until instantiation.
 */
template<class>
struct unit_roundoff_always_false : std::false_type {};

} // namespace detail


/**
 * @brief Trait providing the unit roundoff of a scalar type.
 *
 * The primary template rejects unsupported types. Supported scalar types are
 * provided by the specializations below.
 *
 * @tparam T Scalar type.
 */
template<class T>
struct unit_roundoff_traits {
    /**
     * @brief Rejects an unsupported scalar type at compile time.
     * @return No value; the function cannot be instantiated successfully.
     */
    static T value()
    {
        static_assert(
            detail::unit_roundoff_always_false<T>::value,
            "unit_roundoff_traits: unsupported scalar type"
        );

        return T{}; // Required syntactically; never reached.
    }
};


/**
 * @brief Unit-roundoff trait for HDNUM single precision.
 */
template<>
struct unit_roundoff_traits<hdnum::FP32> {
    /// Returns epsilon/2 in hdnum::FP32.
    static hdnum::FP32 value()
    {
        return std::numeric_limits<hdnum::FP32>::epsilon()
             / hdnum::FP32(2);
    }
};


/**
 * @brief Unit-roundoff trait for HDNUM double precision.
 */
template<>
struct unit_roundoff_traits<hdnum::FP64> {
    /// Returns epsilon/2 in hdnum::FP64.
    static hdnum::FP64 value()
    {
        return std::numeric_limits<hdnum::FP64>::epsilon()
             / hdnum::FP64(2);
    }
};


#ifdef HDNUM_HAS_CPFLOAT

/**
 * @brief Unit-roundoff trait for an HDNUM CPFloat type.
 *
 * @tparam m Significand-precision parameter.
 * @tparam e Exponent-range parameter; it does not affect unit roundoff.
 */
template<int m, int e>
struct unit_roundoff_traits<hdnum::CPFloat<m, e>> {
    using scalar_type = hdnum::CPFloat<m, e>;

    /// Returns 2^(-m) in the CPFloat type.
    static scalar_type value()
    {
        return pow2_neg<scalar_type>(m);
    }
};

#endif


#ifdef HDNUM_HAS_GMP

/**
 * @brief Unit-roundoff trait for a GMP-backed HDNUM scalar type.
 *
 * @tparam m Precision parameter in bits.
 */
template<int m>
struct unit_roundoff_traits<hdnum::FP<m>> {
    using scalar_type = hdnum::FP<m>;

    /// Returns 2^(-m) in the GMP-backed type.
    static scalar_type value()
    {
        return pow2_neg<scalar_type>(m);
    }
};

#endif


/**
 * @brief Returns the default unit roundoff for a supported scalar type.
 *
 * @tparam T Scalar type with a unit_roundoff_traits specialization.
 * @return Unit roundoff represented in T.
 *
 * @note An unsupported type produces a compile-time error.
 */
template<class T>
T default_unit_roundoff()
{
    return unit_roundoff_traits<T>::value();
}

} // namespace mpir