#pragma once

/**
 * @file hdnum_conversions.hpp
 * @brief Scalar and container conversions between numerical precision types.
 *
 * This header provides the conversion utilities used to move values between
 * native, CPFloat, and GMP-backed HDNUM scalar types. T_in denotes the source
 * type and T_out the destination type. Scalars use a direct constructor when
 * available; GMP-backed inputs additionally support conversion through double.
 * Vectors and dense matrices are converted element by element because HDNUM
 * does not provide heterogeneous whole-container conversions.
 */

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "hdnum.hh"

namespace mpir {


/**
 * @brief Dependent false value used in compile-time diagnostics.
 *
 * The unnamed parameter pack delays evaluation until template instantiation.
 */
template<class...>
struct always_false : std::false_type {};


/**
 * @brief Identifies an exact GMP-backed HDNUM scalar type.
 *
 * The primary template is false; the specialization for hdnum::FP<m> is true.
 * Use is_hdnum_fp_v when cv- and reference-qualified types must also be
 * recognized.
 *
 * @tparam T Type to inspect.
 */
template<class T>
struct is_hdnum_fp_impl : std::false_type {};

/**
 * @brief Specialization identifying hdnum::FP<m>.
 *
 * @tparam m GMP precision parameter.
 */
template<int m>
struct is_hdnum_fp_impl<hdnum::FP<m>> : std::true_type {};

/**
 * @brief True if T is hdnum::FP<m>, ignoring cv and reference qualifiers.
 *
 * @tparam T Type to inspect.
 */
template<class T>
inline constexpr bool is_hdnum_fp_v =
    is_hdnum_fp_impl<
        std::remove_cv_t<std::remove_reference_t<T>>
    >::value;


/**
 * @brief Converts one scalar to a destination numerical type.
 *
 * Direct construction of T_out from T_in is preferred. If that is unavailable
 * and T_in is hdnum::FP<m>, the value is extracted as double and used to
 * construct T_out. The fallback is therefore limited to the range and
 * precision of double.
 *
 * @tparam T_out Destination scalar type.
 * @tparam T_in Source scalar type.
 * @param x Source value.
 * @return Converted value in T_out.
 *
 * @note An unsupported type pair produces a compile-time error.
 */
template<class T_out, class T_in>
T_out scalar_cast(const T_in& x)
{
    // Prefer a direct conversion supplied by the scalar types.
    if constexpr (std::is_constructible_v<T_out, const T_in&>) {
        return T_out(x);
    }
    // Bridge GMP-backed inputs through double when no direct path exists.
    else if constexpr (
        is_hdnum_fp_v<T_in> &&
        std::is_constructible_v<T_out, double>
    ) {
        return T_out(x.getNumber().get_d());
    }
    else {
        static_assert(always_false<T_out, T_in>::value,
                      "mpir::scalar_cast: unsupported scalar conversion");
    }
}


/**
 * @brief Converts an HDNUM vector element by element.
 *
 * The destination must already have the same size as the source. Each entry is
 * converted with scalar_cast<T_out>().
 *
 * @tparam T_out Destination scalar type.
 * @tparam T_in Source scalar type.
 * @param[out] out Preallocated destination vector.
 * @param[in] in Source vector.
 *
 * @throws std::invalid_argument If the vector sizes differ.
 */
template<class T_out, class T_in>
void convert(hdnum::Vector<T_out>& out,
             const hdnum::Vector<T_in>& in)
{
    if (out.size() != in.size()) {
        throw std::invalid_argument(
            "mpir::convert(Vector): output vector has wrong size"
        );
    }

    for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = scalar_cast<T_out>(in[i]);
    }
}




/**
 * @brief Converts an HDNUM dense matrix element by element.
 *
 * The destination must already have the same dimensions as the source. Each
 * entry is converted with scalar_cast<T_out>().
 *
 * @tparam T_out Destination scalar type.
 * @tparam T_in Source scalar type.
 * @param[out] out Preallocated destination matrix.
 * @param[in] in Source matrix.
 *
 * @throws std::invalid_argument If the matrix dimensions differ.
 */
template<class T_out, class T_in>
void convert(hdnum::DenseMatrix<T_out>& out,
             const hdnum::DenseMatrix<T_in>& in)
{
    if (out.rowsize() != in.rowsize() || out.colsize() != in.colsize()) {
        throw std::invalid_argument(
            "mpir::convert(DenseMatrix): output matrix has wrong size"
        );
    }

    for (std::size_t i = 0; i < in.rowsize(); ++i) {
        for (std::size_t j = 0; j < in.colsize(); ++j) {
            out[i][j] = scalar_cast<T_out>(in[i][j]);
        }
    }
}


/**
 * @brief Returns a dense matrix converted to T_out.
 *
 * Allocates a matrix with the source dimensions and converts every entry with
 * scalar_cast<T_out>().
 *
 * @tparam T_out Destination scalar type.
 * @tparam T_in Source scalar type.
 * @param in Source matrix.
 * @return Converted matrix with the same dimensions as in.
 */
template<class T_out, class T_in>
hdnum::DenseMatrix<T_out> convert_matrix(const hdnum::DenseMatrix<T_in>& in)
{
    hdnum::DenseMatrix<T_out> out(in.rowsize(), in.colsize());
    convert(out, in);
    return out;
}


/**
 * @brief Returns a vector converted to T_out.
 *
 * Allocates a vector with the source size and converts every entry with
 * scalar_cast<T_out>().
 *
 * @tparam T_out Destination scalar type.
 * @tparam T_in Source scalar type.
 * @param in Source vector.
 * @return Converted vector with the same size as in.
 */
template<class T_out, class T_in>
hdnum::Vector<T_out> convert_vector(const hdnum::Vector<T_in>& in)
{
    hdnum::Vector<T_out> out(in.size());
    convert(out, in);
    return out;
}

} // namespace mpir
