#pragma once

#include <cstddef>
#include <stdexcept>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

namespace mpir {

template<class T_reference, class T_data>
hdnum::Vector<T_reference>
high_precision_solve(
    const hdnum::DenseMatrix<T_data>& A,
    const hdnum::Vector<T_data>& b)
{
    if (A.rowsize() == 0 || A.colsize() == 0) {
        throw std::invalid_argument(
            "high_precision_solve: matrix must be nonempty"
        );
    }

    if (A.rowsize() != A.colsize()) {
        throw std::invalid_argument(
            "high_precision_solve: matrix must be square"
        );
    }

    if (b.size() != A.rowsize()) {
        throw std::invalid_argument(
            "high_precision_solve: incompatible matrix and vector sizes"
        );
    }

    const std::size_t n = A.rowsize();

    // Convert the stored problem to reference precision.
    hdnum::DenseMatrix<T_reference> A_reference =
        convert_matrix<T_reference>(A);

    hdnum::Vector<T_reference> rhs_reference =
        convert_vector<T_reference>(b);

    // Full-pivoting LU:
    //
    //     P A Q = L U
    //
    // lr_fullpivot overwrites A_reference with the combined
    // L and U factors.
    hdnum::Vector<std::size_t> p(n);
    hdnum::Vector<std::size_t> q(n);

    hdnum::lr_fullpivot(A_reference, p, q);

    // Apply the row permutation:
    //
    //     L U Q^{-1} x = P b
    hdnum::permute_forward(p, rhs_reference);

    // Solve:
    //
    //     L y = P b
    //
    // rhs_reference is overwritten with y.
    hdnum::solveL(
        A_reference,
        rhs_reference,
        rhs_reference
    );

    // Solve:
    //
    //     U z = y
    hdnum::Vector<T_reference> x_reference(n);

    hdnum::solveR(
        A_reference,
        x_reference,
        rhs_reference
    );

    // Since z = Q^{-1}x, undo the column permutation.
    hdnum::permute_backward(q, x_reference);

    return x_reference;
}

} // namespace mpir