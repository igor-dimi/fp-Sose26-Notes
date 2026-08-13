#pragma once

/**
 * @file reference_solve.hpp
 * @brief Direct full-pivoting LU solves in a selected reference precision.
 *
 * This header provides the direct solve used to construct reference solutions
 * for the numerical experiments. The coefficient matrix and right-hand side
 * are stored in T_data, converted to T_reference, and then factorized and
 * solved entirely in T_reference. The input system is not modified.
 */

#include <cstddef>
#include <stdexcept>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

namespace mpir {

/**
 * @brief Solves `A x = b` by full-pivoting LU in `T_reference`.
 *
 * Converts `A` and `b` from `T_data` to `T_reference`, then performs the
 * factorization, permutations, and triangular solves in `T_reference`.
 * The inputs are unchanged. Use equal template types for a direct solve in the
 * data precision.
 *
 * The routine solves the supplied stored system; increasing T_reference does
 * not recover information already lost when `A` or `b` were stored in
 * `T_data`.
 *
 * @tparam T_reference Arithmetic and result precision.
 * @tparam T_data Input storage precision.
 * @param A Nonempty square coefficient matrix.
 * @param b Right-hand side of compatible size.
 * @return Solution in `T_reference`.
 *
 * @throws std::invalid_argument If `A` is empty or nonsquare, or if its size
 *         is incompatible with `b`.
 *
 * @note Finiteness and singularity are not checked explicitly; failures follow
 *       the behavior of HDNUM's factorization and solve routines.
 */
template<class T_reference, class T_data>
[[nodiscard]]
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

    // Convert the system to the arithmetic precision.
    hdnum::DenseMatrix<T_reference> A_reference =
        convert_matrix<T_reference>(A);

    hdnum::Vector<T_reference> rhs_reference =
        convert_vector<T_reference>(b);

    // Compute P A Q = L U in place.
    hdnum::Vector<std::size_t> p(n);
    hdnum::Vector<std::size_t> q(n);

    hdnum::lr_fullpivot(A_reference, p, q);

    // Apply P and solve L y = P b in place.
    hdnum::permute_forward(p, rhs_reference);
    hdnum::solveL(
        A_reference,
        rhs_reference,
        rhs_reference
    );

    // Solve U z = y.
    hdnum::Vector<T_reference> x_reference(n);

    hdnum::solveR(
        A_reference,
        x_reference,
        rhs_reference
    );

    // Recover x = Q z.
    hdnum::permute_backward(q, x_reference);

    return x_reference;
}

} // namespace mpir
