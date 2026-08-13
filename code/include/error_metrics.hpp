#pragma once

/**
 * @file error_metrics.hpp
 * @brief Infinity-norm error measures for linear-system experiments.
 *
 * This header provides vector and matrix infinity norms, relative forward
 * error, and normwise backward error. All arithmetic used to evaluate a metric
 * is performed in the independently selected T_measure precision after
 * componentwise conversion of the inputs. Scalar error results are converted
 * to double for experiment output.
 */

#include <cstddef>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

namespace mpir {


/**
 * @brief Returns the absolute value of a scalar.
 *
 * This comparison-based implementation also supports HDNUM scalar types that
 * do not provide a suitable std::abs overload.
 *
 * @tparam T Scalar type supporting construction from zero, comparison, and
 *         subtraction.
 * @param x Input value.
 * @return Absolute value of x in type T.
 */
template<class T>
T abs_value(T x) {
    const T zero = T(0);
    return x < zero ? zero - x : x;
}


/**
 * @brief Computes the vector infinity norm in T_measure.
 *
 * Evaluates
 *
 *     ||x||_inf = max_i |x_i|
 *
 * after converting each component to T_measure. The norm of an empty vector
 * is zero.
 *
 * @tparam T_measure Arithmetic and result precision of the norm.
 * @tparam T_x Storage type of the vector entries.
 * @param x Input vector.
 * @return Infinity norm of x in T_measure.
 */
template<class T_measure, class T_x>
T_measure vector_norm_inf(const hdnum::Vector<T_x>& x)
{
    T_measure max_abs = T_measure(0);

    for (std::size_t i = 0; i < x.size(); ++i) {
        const T_measure xi_abs = abs_value(scalar_cast<T_measure>(x[i]));

        if (xi_abs > max_abs) {
            max_abs = xi_abs;
        }
    }

    return max_abs;
}


/**
 * @brief Computes the matrix infinity norm in T_measure.
 *
 * Evaluates
 *
 *     ||A||_inf = max_i sum_j |a_ij|
 *
 * after converting each entry to T_measure. The norm is zero if the matrix has
 * no rows or no columns.
 *
 * @tparam T_measure Arithmetic and result precision of the norm.
 * @tparam T_A Storage type of the matrix entries.
 * @param A Input matrix.
 * @return Infinity norm of A in T_measure.
 */
template<class T_measure, class T_A>
T_measure matrix_norm_inf(const hdnum::DenseMatrix<T_A>& A)
{
    T_measure max_row_sum = T_measure(0);

    for (std::size_t i = 0; i < A.rowsize(); ++i) {
        T_measure row_sum = T_measure(0);

        for (std::size_t j = 0; j < A.colsize(); ++j) {
            row_sum += abs_value(scalar_cast<T_measure>(A[i][j]));
        }

        if (row_sum > max_row_sum) {
            max_row_sum = row_sum;
        }
    }

    return max_row_sum;
}


/**
 * @brief Computes the relative forward error in the infinity norm.
 *
 * Evaluates
 *
 *     ||x - x_ref||_inf / ||x_ref||_inf
 *
 * in T_measure and converts the result to double. If the reference norm is
 * zero, the function returns the absolute error ||x - x_ref||_inf instead.
 *
 * @tparam T_measure Arithmetic precision used to evaluate the error.
 * @tparam T_x Storage type of the approximate solution.
 * @tparam T_ref Storage type of the reference solution.
 * @param x Approximate solution.
 * @param x_ref Reference solution.
 * @return Relative forward error, or the absolute forward error if x_ref is
 *         zero.
 *
 * @pre x and x_ref have equal sizes.
 */
template<class T_measure, class T_x, class T_ref>
double relative_forward_error_inf(
    const hdnum::Vector<T_x>& x,
    const hdnum::Vector<T_ref>& x_ref
)
{
    const std::size_t n = x.size();

    hdnum::Vector<T_measure> diff(n);

    for (std::size_t i = 0; i < n; ++i) {
        diff[i] = scalar_cast<T_measure>(x[i])
                - scalar_cast<T_measure>(x_ref[i]);
    }

    const T_measure numerator =
        vector_norm_inf<T_measure>(diff);

    const T_measure denominator =
        vector_norm_inf<T_measure>(x_ref);

    // Avoid division by zero for a zero reference solution.
    if (denominator == T_measure(0)) {
        return scalar_cast<double>(numerator);
    }

    return scalar_cast<double>(numerator / denominator);
}


/**
 * @brief Computes the normwise relative backward error in the infinity norm.
 *
 * Evaluates
 *
 *     ||b - A*x||_inf
 *     -------------------------------
 *     ||A||_inf ||x||_inf + ||b||_inf
 *
 * entirely in T_measure and converts the result to double. If the denominator
 * is zero, the function returns the unnormalized residual norm.
 *
 * @tparam T_measure Arithmetic precision used to evaluate the error.
 * @tparam T_A Storage type of the matrix entries.
 * @tparam T_b Storage type of the right-hand-side entries.
 * @tparam T_x Storage type of the approximate-solution entries.
 * @param A Coefficient matrix.
 * @param b Right-hand side.
 * @param x Approximate solution.
 * @return Normwise backward error, or the residual infinity norm if the
 *         denominator is zero.
 *
 * @pre A is square, and its dimension equals the sizes of b and x.
 */
template<class T_measure, class T_A, class T_b, class T_x>
double normwise_backward_error_inf(
    const hdnum::DenseMatrix<T_A>& A,
    const hdnum::Vector<T_b>& b,
    const hdnum::Vector<T_x>& x
)
{
    const std::size_t n = b.size();

    hdnum::Vector<T_measure> residual(n);

    // Form b - A*x directly in measurement precision.
    for (std::size_t i = 0; i < n; ++i) {
        T_measure Ax_i = T_measure(0);

        for (std::size_t j = 0; j < n; ++j) {
            Ax_i += scalar_cast<T_measure>(A[i][j])
                  * scalar_cast<T_measure>(x[j]);
        }

        residual[i] = scalar_cast<T_measure>(b[i]) - Ax_i;
    }

    const T_measure numerator =
        vector_norm_inf<T_measure>(residual);

    const T_measure norm_A =
        matrix_norm_inf<T_measure>(A);

    const T_measure norm_x =
        vector_norm_inf<T_measure>(x);

    const T_measure norm_b =
        vector_norm_inf<T_measure>(b);

    const T_measure denominator = norm_A * norm_x + norm_b;

    // Avoid division by zero when the normalization vanishes.
    if (denominator == T_measure(0)) {
        return scalar_cast<double>(numerator);
    }

    return scalar_cast<double>(numerator / denominator);
}


} // namespace mpir
