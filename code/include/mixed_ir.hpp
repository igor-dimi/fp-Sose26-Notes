#pragma once

/**
 * @file mixed_ir.hpp
 * @brief Three-precision LU iterative refinement for linear systems.
 *
 * This header implements mixed-precision iterative refinement for
 *
 *     A*x = b.
 *
 * The LU factorization and correction solves use T_factor, the system data
 * and iterates use T_work, and residuals are computed in T_residual. The
 * implementation also provides optional residual scaling, convergence and
 * divergence diagnostics, and non-finite-value checks.
 */

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"
#include "unit_roundoff.hpp"

namespace mpir {


/**
 * @brief Termination status of mixed-precision iterative refinement.
 */
enum class MixedIRStatus {
    converged,                     ///< The convergence criterion was satisfied.
    max_iterations,                ///< The maximum number of iterations was reached.
    factorization_input_non_finite,///< A or b became NaN/Inf in factor precision.
    non_finite,                    ///< NaN or Inf appeared later in the algorithm.
    diverged                       ///< Rapid numerical growth indicated divergence.
};


/**
 * @brief Configuration options for mixed-precision iterative refinement.
 *
 * @tparam T_work Working-precision scalar type.
 */
template<class T_work>
struct MixedIROptions {
    /// Maximum number of refinement updates.
    std::size_t max_iterations = 10;

    /// Relative-correction tolerance. A value <= 0 selects u_work.
    T_work rel_correction_tol = T_work(0);

    /// Enable detection of persistent correction growth.
    bool detect_divergence = true;

    /// Growth factor in ||d_k||_2 > factor * ||d_{k-1}||_2.
    T_work divergence_growth_factor = T_work(10);

    /// Consecutive growth steps required to report divergence.
    std::size_t divergence_growth_steps = 3;

    /// Store x0, x1, ... for convergence-history experiments.
    bool store_iterates = false;

    /// Scale residuals by their infinity norm before conversion to T_factor.
    bool scale_residual = false;

    /// Record diagnostics for residual narrowing to T_factor.
    bool record_residual_diagnostics = false;
};


/**
 * @brief Diagnostics for narrowing one residual to factor precision.
 */
struct ResidualDiagnostic {
    /// Refinement index k for r_k = b - A*x_k.
    std::size_t iteration = 0;

    /// Infinity norm of the original residual.
    double residual_inf_norm = 0.0;

    /// Smallest nonzero absolute component of the original residual.
    double min_nonzero_abs = 0.0;

    /// Number of nonzero components in the original residual.
    std::size_t nonzero_components = 0;

    /// Nonzero correction-RHS components rounded to zero in T_factor.
    std::size_t zeroed_by_conversion = 0;
};


/**
 * @brief Result and optional diagnostics from mixed-precision refinement.
 *
 * @tparam T_work Working-precision scalar type.
 */
template<class T_work>
struct MixedIRResult {
    /// Final accepted iterate in working precision.
    hdnum::Vector<T_work> x;

    /// Number of completed refinement updates; the initial solve is excluded.
    std::size_t iterations = 0;

    /// Reason why the algorithm terminated.
    MixedIRStatus status = MixedIRStatus::max_iterations;

    /**
     * @brief Returns true iff the termination status is converged.
     */
    [[nodiscard]] bool converged() const noexcept
    {
        return status == MixedIRStatus::converged;
    }

    /// Relative correction of the last completed update.
    double final_rel_correction = 0.0;

    /// Relative correction ||d_k||_2 / ||x_k||_2 for each completed update.
    std::vector<double> rel_corrections;

    /**
     * @brief Absolute correction norms ||d_k||_2.
     *
     * A divergence-triggering correction is recorded even if it is not
     * applied to the iterate.
     */
    std::vector<double> correction_norms;

    /**
     * @brief Optional iterate history x0, x1, ... .
     *
     * When enabled, iterates.size() == iterations + 1.
     */
    std::vector<hdnum::Vector<T_work>> iterates;

    /**
     * @brief Optional residual-conversion diagnostics.
     *
     * With residual scaling, magnitude statistics refer to the original
     * residual while zeroed_by_conversion refers to the normalized one.
     */
    std::vector<ResidualDiagnostic> residual_diagnostics;
};


/**
 * @brief Computes the relative correction ||d||_2 / ||x||_2 in type T.
 *
 * If ||x||_2 is zero, the denominator is replaced by 1.
 *
 * @param d Correction vector.
 * @param x Current iterate.
 * @return Relative correction in type T.
 *
 * @throws std::invalid_argument if d and x have different sizes.
 */
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


/**
 * @brief Solves A*x = b from a full-pivoting LU factorization.
 *
 * Uses the factorization P*A*Q = L*U stored in HDNUM's combined LU form.
 *
 * @param LU Combined L/U factors.
 * @param p Row-permutation data.
 * @param q Column-permutation data.
 * @param b Right-hand side.
 * @return Solution vector x.
 *
 * @throws std::invalid_argument if matrix, vector, or permutation sizes
 *         are incompatible.
 */
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


/**
 * @brief Computes r = b - A*x in residual precision.
 *
 * A, b, and x are converted from T_work to T_residual before evaluation.
 *
 * @tparam T_residual Residual-precision scalar type.
 * @tparam T_work Working-precision scalar type.
 * @param A System matrix.
 * @param b Right-hand side.
 * @param x Current iterate.
 * @return Residual in T_residual.
 *
 * @throws std::invalid_argument if dimensions are incompatible.
 */
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
 * @brief Checks whether every vector entry is finite.
 *
 * IEEE-like and CPFloat values are tested after conversion to double.
 * GMP-backed HDNUM FP values are treated as finite because the type has
 * no NaN or infinity representation.
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
 * @brief Checks whether every dense-matrix entry is finite.
 *
 * GMP-backed HDNUM FP values are treated as finite.
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
 * GMP-backed HDNUM FP values are treated as finite.
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

/**
 * @brief Solves A*x = b by three-precision LU iterative refinement.
 *
 * The algorithm computes a full-pivoting LU factorization in T_factor,
 * stores and updates the solution in T_work, and evaluates residuals in
 * T_residual. The same LU factors are reused for all correction solves.
 *
 * Refinement stops when
 *
 *     ||d_k||_2 / ||x_k||_2 < tol,
 *
 * where tol defaults to the unit roundoff of T_work, or when another
 * termination condition is reached. Optional residual scaling normalizes
 * r_k before conversion to T_factor and rescales the correction in T_work.
 *
 * @tparam T_factor Factorization and correction-solve scalar type.
 * @tparam T_work System-data, iterate, and update scalar type.
 * @tparam T_residual Residual-computation scalar type.
 * @param A Square nonempty system matrix in working precision.
 * @param b Right-hand side in working precision.
 * @param options Algorithm and diagnostic options.
 * @return Final solution, termination status, and requested diagnostics.
 *
 * @throws std::invalid_argument if A is empty, nonsquare, or incompatible
 *         with b.
 */
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

    // Conversion to T_factor may overflow; reject non-finite input.
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

    // Reject non-finite values produced by factorization or the initial solve.
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

    result.correction_norms.reserve(
        options.max_iterations
    );

    if (options.record_residual_diagnostics) {
        result.residual_diagnostics.reserve(
            options.max_iterations
        );
    }

    if (options.store_iterates) {
        result.iterates.reserve(
            options.max_iterations + 1
        );

        // iterates[0] = x0.
        result.iterates.push_back(result.x);
    }


    // State for persistent correction-growth detection.
    T_work previous_correction_norm = T_work(0);
    bool have_previous_correction_norm = false;
    std::size_t consecutive_growth_steps = 0;


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

        // Reject a non-finite residual.
        if (!all_finite(r_r)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }


        // Compute the scaling norm and, if requested, residual statistics.
        T_residual residual_inf_norm_r = T_residual(0);
        T_residual min_nonzero_abs_r = T_residual(0);
        std::size_t nonzero_components = 0;

        if (options.scale_residual ||
            options.record_residual_diagnostics) {
            using std::abs;

            for (std::size_t i = 0; i < n; ++i) {
                const T_residual abs_value = abs(r_r[i]);

                if (abs_value > residual_inf_norm_r) {
                    residual_inf_norm_r = abs_value;
                }

                if (options.record_residual_diagnostics &&
                    abs_value > T_residual(0)) {
                    if (nonzero_components == 0 ||
                        abs_value < min_nonzero_abs_r) {
                        min_nonzero_abs_r = abs_value;
                    }

                    ++nonzero_components;
                }
            }
        }

        // If enabled, normalize r_k so every component has magnitude <= 1.
        hdnum::Vector<T_residual> correction_rhs_r(r_r);
        T_residual residual_scale_r = T_residual(1);
        bool residual_was_scaled = false;

        if (options.scale_residual &&
            residual_inf_norm_r > T_residual(0)) {
            residual_scale_r = residual_inf_norm_r;
            residual_was_scaled = true;

            for (std::size_t i = 0; i < n; ++i) {
                correction_rhs_r[i] =
                    correction_rhs_r[i] / residual_scale_r;
            }
        }

        // Narrow the correction right-hand side to factor precision.
        hdnum::Vector<T_factor> r_f(n);
        convert(r_f, correction_rhs_r);

        // Check both the unscaled path and custom scalar types defensively.
        if (!all_finite(r_f)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }

        if (options.record_residual_diagnostics) {
            std::size_t zeroed_by_conversion = 0;

            for (std::size_t i = 0; i < n; ++i) {
                if (!(correction_rhs_r[i] == T_residual(0)) &&
                    r_f[i] == T_factor(0)) {
                    ++zeroed_by_conversion;
                }
            }

            ResidualDiagnostic diagnostic;
            diagnostic.iteration = k;
            diagnostic.residual_inf_norm =
                scalar_cast<double>(residual_inf_norm_r);
            diagnostic.min_nonzero_abs =
                nonzero_components > 0
                    ? scalar_cast<double>(min_nonzero_abs_r)
                    : 0.0;
            diagnostic.nonzero_components =
                nonzero_components;
            diagnostic.zeroed_by_conversion =
                zeroed_by_conversion;

            result.residual_diagnostics.push_back(
                diagnostic
            );
        }

        // Reuse the low-precision LU factors for the correction solve.
        hdnum::Vector<T_factor> d_f =
            solve_with_lu_fullpivot(
                A_f,
                p,
                q,
                r_f
            );

        
        // Reject a non-finite correction from the triangular solve.
        if (!all_finite(d_f)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }

        // Convert the normalized correction to working precision.
        hdnum::Vector<T_work> d_w(n);
        convert(d_w, d_f);

        // Defensively validate the converted working-precision correction.
        if (!all_finite(d_w)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }

        // Undo residual scaling in T_work: d_k = theta_k * d_hat_k.
        if (residual_was_scaled) {
            const T_work residual_scale_w =
                scalar_cast<T_work>(residual_scale_r);

            if (!scalar_is_finite(residual_scale_w)) {
                result.status = MixedIRStatus::non_finite;
                return result;
            }

            for (std::size_t i = 0; i < n; ++i) {
                d_w[i] = residual_scale_w * d_w[i];
            }

            if (!all_finite(d_w)) {
                result.status = MixedIRStatus::non_finite;
                return result;
            }
        }

        // Compute ||d_k||_2 for divergence detection.
        const T_work correction_norm_w = 
            hdnum::norm(d_w);

        const double correction_norm =
        scalar_cast<double>(
            correction_norm_w
        );


        // Reject overflow in the norm or its diagnostic conversion.
        if (!std::isfinite(correction_norm)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }


        // Record the norm before divergence detection.
        result.correction_norms.push_back(
            correction_norm
        );


        // Stop after the configured number of consecutive growth steps.
        if (options.detect_divergence) {
            if (have_previous_correction_norm) {
                const bool excessive_growth =
                    previous_correction_norm > T_work(0)
                    &&
                    correction_norm_w
                        > options.divergence_growth_factor
                            * previous_correction_norm;

                if (excessive_growth) {
                    ++consecutive_growth_steps;
                } else {
                    consecutive_growth_steps = 0;
                }

                if (consecutive_growth_steps
                    >= options.divergence_growth_steps) {
                    result.status = MixedIRStatus::diverged;
                    return result;
                }
            }

            previous_correction_norm =
                correction_norm_w;

            have_previous_correction_norm = true;
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
        
        // The norm calculation can overflow even for finite entries.
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

        // Accept the candidate update only if every entry remains finite.
        if (!all_finite(x_next)) {
            result.status = MixedIRStatus::non_finite;
            return result;
        }

        // The update was successful, so commit x_{k + 1}
        result.x = x_next;

        // Record diagnostics only after a successful update.
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