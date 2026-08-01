#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"
#include "unit_roundoff.hpp"

namespace mpir {


/**
 * @brief Describes why the mixed-precision iterative-refinement
 *        algorithm terminated.
 *
 * The status distinguishes successful convergence, exhaustion of the
 * iteration limit, non-finite factorization input, non-finite values
 * produced during the algorithm, and detected divergence.
 */
enum class MixedIRStatus {
    converged,                     ///< The convergence criterion was satisfied.
    max_iterations,                ///< The maximum number of iterations was reached.
    factorization_input_non_finite,///< A or b became NaN/Inf in factor precision.
    non_finite,                    ///< NaN or Inf appeared later in the algorithm.
    diverged                       ///< Rapid numerical growth indicated divergence.
};


template<class T_work>
struct MixedIROptions {
    std::size_t max_iterations = 10;

    // A value <= 0 means that the unit roundoff of T_work is used.
    T_work rel_correction_tol = T_work(0);

    // Store x0, x1, x2, ... for convergence-history experiments.
    // This can be disabled for large parameter sweeps to save memory.
    bool store_iterates = false;
};


template<class T_work>
struct MixedIRResult {
    // Final approximate solution in working precision.
    hdnum::Vector<T_work> x;

    // Number of refinement updates performed.
    // The initial low-precision solve for x0 is not counted.
    std::size_t iterations = 0;

    /**
     * @brief Reason why the algorithm terminated.
     */
    MixedIRStatus status = MixedIRStatus::max_iterations;

    /**
     * @brief Returns whether iterative refinement converged successfully.
     *
     * Convergence is derived from the termination status, avoiding a
     * separate Boolean state that could become inconsistent with it.
     */
    [[nodiscard]] bool converged() const noexcept
    {
        return status == MixedIRStatus::converged;
    }

    // Last computed value of
    //
    //     ||d_k||_2 / ||x_k||_2.
    //
    // Stored as double for diagnostics and CSV output.
    double final_rel_correction = 0.0;

    // One entry for every completed refinement update.
    //
    // rel_corrections[k] is the relative correction that
    // produced x_{k+1} from x_k.
    std::vector<double> rel_corrections;

    // Optional iterate history.
    //
    // iterates[0] = x0
    // iterates[1] = x1
    // ...
    //
    // If enabled:
    //
    //     iterates.size() == iterations + 1.
    std::vector<hdnum::Vector<T_work>> iterates;
};


// Compute
//
//     ||d||_2 / ||x||_2
//
// using arithmetic in T.
//
// If ||x||_2 is zero, use 1 as the denominator.
template<class T>
T relative_correction_2(
    const hdnum::Vector<T>& d,
    const hdnum::Vector<T>& x)
{
    if (d.size() != x.size()) {
        throw std::invalid_argument(
            "relative_correction_2: vector sizes do not match"
        );
    }

    const T norm_d = hdnum::norm(d);
    const T norm_x = hdnum::norm(x);

    const T denominator =
        norm_x > T(0) ? norm_x : T(1);

    return norm_d / denominator;
}


// Solve A*x = b using an already computed full-pivoting
// LU factorization
//
//     P*A*Q = L*U.
//
// LU contains the combined L and U factors. The vectors p and q
// contain the row and column permutation information.
template<class T>
hdnum::Vector<T>
solve_with_lu_fullpivot(
    const hdnum::DenseMatrix<T>& LU,
    const hdnum::Vector<std::size_t>& p,
    const hdnum::Vector<std::size_t>& q,
    const hdnum::Vector<T>& b)
{
    const std::size_t n = b.size();

    if (LU.rowsize() != LU.colsize() ||
        LU.rowsize() != n) {
        throw std::invalid_argument(
            "solve_with_lu_fullpivot: incompatible matrix and vector sizes"
        );
    }

    if (p.size() != n || q.size() != n) {
        throw std::invalid_argument(
            "solve_with_lu_fullpivot: invalid permutation-vector sizes"
        );
    }

    hdnum::Vector<T> y(b);
    hdnum::Vector<T> x(n);

    // Form P*b.
    hdnum::permute_forward(p, y);

    // Solve L*y = P*b in place.
    hdnum::solveL(LU, y, y);

    // Solve U*x = y.
    hdnum::solveR(LU, x, y);

    // Undo the column permutation.
    hdnum::permute_backward(q, x);

    return x;
}


// Compute
//
//     r = b - A*x
//
// in residual precision T_residual.
//
// A, b, and x are stored in working precision and are converted
// to residual precision before the matrix-vector product.
template<class T_residual, class T_work>
hdnum::Vector<T_residual>
compute_residual(
    const hdnum::DenseMatrix<T_work>& A,
    const hdnum::Vector<T_work>& b,
    const hdnum::Vector<T_work>& x)
{
    if (A.rowsize() != b.size() ||
        A.colsize() != x.size()) {
        throw std::invalid_argument(
            "compute_residual: incompatible matrix and vector sizes"
        );
    }

    hdnum::DenseMatrix<T_residual> A_r(
        A.rowsize(),
        A.colsize()
    );

    hdnum::Vector<T_residual> b_r(b.size());
    hdnum::Vector<T_residual> x_r(x.size());
    hdnum::Vector<T_residual> Ax_r(b.size());

    convert(A_r, A);
    convert(b_r, b);
    convert(x_r, x);

    A_r.mv(Ax_r, x_r);

    return b_r - Ax_r;
}

/**
 * @brief Checks whether every entry of a vector is finite.
 *
 * For ordinary IEEE types and CPFloat, values are converted to double
 * and tested for NaN or infinity.
 *
 * HDNUM's GMP-backed FP type does not represent NaN or infinity, so
 * every successfully constructed FP value is finite.
 */
template<class T>
bool all_finite(const hdnum::Vector<T>& v)
{
    if constexpr (is_hdnum_fp_v<T>) {
        return true;
    } else {
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (!std::isfinite(static_cast<double>(v[i]))) {
                return false;
            }
        }

        return true;
    }
}


/**
 * @brief Checks whether every entry of a dense matrix is finite.
 */
template<class T>
bool all_finite(const hdnum::DenseMatrix<T>& A)
{
    if constexpr (is_hdnum_fp_v<T>) {
        return true;
    } else {
        for (std::size_t i = 0; i < A.rowsize(); ++i) {
            for (std::size_t j = 0; j < A.colsize(); ++j) {
                if (!std::isfinite(
                        static_cast<double>(A[i][j]))) {
                    return false;
                }
            }
        }

        return true;
    }
}


/**
 * @brief Checks whether a scalar value is finite.
 *
 * GMP-backed FP values cannot represent NaN or infinity, so every
 * successfully constructed value of that type is treated as finite.
 */
template<class T>
bool scalar_is_finite(const T& value)
{
    if constexpr (is_hdnum_fp_v<T>) {
        return true;
    } else {
        return std::isfinite(
            static_cast<double>(value)
        );
    }
}

// Three-precision iterative refinement.
//
// T_factor:
//     precision used for LU factorization and correction solves.
//
// T_work:
//     precision in which A, b, and the iterates are stored.
//
// T_residual:
//     precision used to compute the residual.
template<class T_factor, class T_work, class T_residual>
MixedIRResult<T_work>
mixed_ir(
    const hdnum::DenseMatrix<T_work>& A,
    const hdnum::Vector<T_work>& b,
    const MixedIROptions<T_work>& options = {})
{
    if (A.rowsize() == 0 ||
        A.rowsize() != A.colsize()) {
        throw std::invalid_argument(
            "mixed_ir: A must be square and nonempty"
        );
    }

    if (A.rowsize() != b.size()) {
        throw std::invalid_argument(
            "mixed_ir: incompatible matrix and right-hand-side sizes"
        );
    }

    const std::size_t n = b.size();

    const T_work tol =
        options.rel_correction_tol > T_work(0)
            ? options.rel_correction_tol
            : default_unit_roundoff<T_work>();

    MixedIRResult<T_work> result;

    // 1. Convert A and b to factorization precision.
    hdnum::DenseMatrix<T_factor> A_f(n, n);
    hdnum::Vector<T_factor> b_f(n);

    convert(A_f, A);
    convert(b_f, b);

    // The conversion may overflow if A or b exceeds the numerical
    // range of T_factor. In that case, do not attempt LU factorization.
    if (!all_finite(A_f) || !all_finite(b_f)) {
        result.status =
            MixedIRStatus::factorization_input_non_finite;

        return result;
    }

    // 2. Compute the full-pivoting LU factorization
    //
    //     P*A_f*Q = L*U.
    //
    // lr_fullpivot overwrites A_f with the combined L/U factors.
    hdnum::Vector<std::size_t> p(n);
    hdnum::Vector<std::size_t> q(n);

    hdnum::lr_fullpivot(A_f, p, q);


    // 3. Compute the initial solution x0 in factorization precision.
    hdnum::Vector<T_factor> x0_f =
        solve_with_lu_fullpivot(A_f, p, q, b_f);

    // LU factorization or the triangular solve may produce NaN or Inf,
    // even when the original factorization inputs were finite.
    if (!all_finite(x0_f)) {
        result.status = MixedIRStatus::non_finite;
        return result;
    }


    // 4. Convert the validated initial solution x0_f to working precision.
    result.x = hdnum::Vector<T_work>(n);
    convert(result.x, x0_f);

    result.rel_corrections.reserve(
        options.max_iterations
    );

    if (options.store_iterates) {
        result.iterates.reserve(
            options.max_iterations + 1
        );

        // iterates[0] = x0.
        result.iterates.push_back(result.x);
    }


    // 5. Refinement loop.
    for (std::size_t k = 0;
         k < options.max_iterations;
         ++k) {

        // Compute
        //
        //     r_k = b - A*x_k
        //
        // in residual precision.
        hdnum::Vector<T_residual> r_r =
            compute_residual<T_residual>(
                A,
                b,
                result.x
            );

        // Useful when T_residual is an IEEE-like type.
        if (!all_finite(r_r)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }


        // The triangular correction solve uses T_factor LU
        // factors, so its right-hand side must be in T_factor.
        hdnum::Vector<T_factor> r_f(n);
        convert(r_f, r_r);

        // Important: narrowing a finite residual to T_factor can overflow.
        if (!all_finite(r_f)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }


        // Solve
        //
        //     A*d_k = r_k
        //
        // using the existing low-precision LU factors.
        hdnum::Vector<T_factor> d_f =
            solve_with_lu_fullpivot(
                A_f,
                p,
                q,
                r_f
            );

        
        // The low-precision triangular solve may produce NaN or Inf,
        // for example because of invalid LU factors or a zero pivot.
        if (!all_finite(d_f)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }

        // Convert the correction to working precision.
        hdnum::Vector<T_work> d_w(n);
        convert(d_w, d_f);

        // This is mostly defensive because T_work is normally at least as
        // capable as T_factor.
        if (!all_finite(d_w)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }



        // Compute
        //
        //     ||d_k||_2 / ||x_k||_2
        //
        // before updating x_k.
        const T_work rel_correction_w =
            relative_correction_2(
                d_w,
                result.x
            );
        
        // Although d_w and result.x contain finite entries, the norm
        // calculation itself can overflow, for example while forming
        // sums of squares.
        if (!scalar_is_finite(rel_correction_w)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }


        const double rel_correction =
            scalar_cast<double>(
                rel_correction_w
            );

        // Form a candidate for
        //
        //     x_{k+1} = x_k + d_k
        //
        // without overwriting the last valid iterate.
        hdnum::Vector<T_work> x_next(result.x);

        for (std::size_t i = 0; i < n; ++i) {
            x_next[i] += d_w[i];
        }

        // Two finite values can produce infinity when their sum exceeds the
        // range of T_work. Only accept the update if every entry is finite.
        if (!all_finite(x_next)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }

        // The update was successful, so commit x_{k + 1}
        result.x = x_next;

        // record diagnostics only for a successfully completed update
        result.rel_corrections.push_back(
            rel_correction
        );

        result.final_rel_correction =
            rel_correction;




        // One refinement update has now been completed.
        result.iterations = k + 1;


        // Store x_{k+1}.
        if (options.store_iterates) {
            result.iterates.push_back(
                result.x
            );
        }


        // Stop after completing the update associated with d_k.
        if (rel_correction_w < tol) {
            result.status = MixedIRStatus::converged;
            break;
        }
    }

    return result;
}


} // namespace mpir