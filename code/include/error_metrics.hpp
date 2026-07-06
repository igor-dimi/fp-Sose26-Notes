#pragma once

#include <cstddef>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

namespace mpir {


template<class T>
T abs_value(T x) {
    const T zero = T(0);
    return x < zero ? zero - x : x;
}

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

    if (denominator == T_measure(0)) {
        return scalar_cast<double>(numerator);
    }

    return scalar_cast<double>(numerator / denominator);
}

template<class T_measure, class T_A, class T_b, class T_x>
double normwise_backward_error_inf(
    const hdnum::DenseMatrix<T_A>& A,
    const hdnum::Vector<T_b>& b,
    const hdnum::Vector<T_x>& x
)
{
    const std::size_t n = b.size();

    hdnum::Vector<T_measure> residual(n);

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

    if (denominator == T_measure(0)) {
        return scalar_cast<double>(numerator);
    }

    return scalar_cast<double>(numerator / denominator);
}


}