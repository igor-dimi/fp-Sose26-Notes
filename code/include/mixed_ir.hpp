#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"
#include "unit_roundoff.hpp"

namespace mpir {


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

    // True if the relative-correction stopping criterion was
    // satisfied during the refinement loop.
    bool converged = false;

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


    // 1. Convert A and b to factorization precision.
    hdnum::DenseMatrix<T_factor> A_f(n, n);
    hdnum::Vector<T_factor> b_f(n);

    convert(A_f, A);
    convert(b_f, b);


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


    // 4. Convert x0 to working precision.
    MixedIRResult<T_work> result;

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


        // The triangular correction solve uses T_factor LU
        // factors, so its right-hand side must be in T_factor.
        hdnum::Vector<T_factor> r_f(n);
        convert(r_f, r_r);


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


        // Convert the correction to working precision.
        hdnum::Vector<T_work> d_w(n);
        convert(d_w, d_f);


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

        const double rel_correction =
            scalar_cast<double>(
                rel_correction_w
            );

        result.rel_corrections.push_back(
            rel_correction
        );

        result.final_rel_correction =
            rel_correction;


        // Form
        //
        //     x_{k+1} = x_k + d_k
        //
        // in working precision.
        for (std::size_t i = 0; i < n; ++i) {
            result.x[i] += d_w[i];
        }


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
            result.converged = true;
            break;
        }
    }

    return result;
}


} // namespace mpir