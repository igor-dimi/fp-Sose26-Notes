#pragma once

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"
#include "unit_roundoff.hpp"


namespace mpir {

struct MixedIROptions {
    std::size_t max_iterations = 10;
    double rel_correction_tol = 0.0;

    // Store x0, x1, x2, ... for convergence-history experiments.
    // Keep false for large condition sweeps if memory becomes annoying.
    bool store_iterates = false;
};

template<class T_work>
struct MixedIRResult {
    // Final approximate solution, stored in the working precision T_work.
    hdnum::Vector<T_work> x;

    // Number of refinement iterations actually performed.
    // This does not count the initial LU solve for x0.
    std::size_t iterations = 0;

    // True if the stopping criterion was reached before max_iterations.
    // False means the algorithm stopped because the iteration limit was reached.
    bool converged = false;

    // Last computed relative correction norm ||d_i|| / ||x_i||.
    // Useful for diagnostics, especially if converged == false.
    double final_rel_correction = 0.0;

    // History of relative correction norms, one entry per refinement iteration.
    // Useful for convergence diagnostics and later plotting.
    std::vector<double> rel_corrections;

    // Optional history of iterates.
    // iterates[0] is x0 after the initial low-precision solve.
    // iterates[k] is the solution after k refinement steps.
    std::vector<hdnum::Vector<T_work>> iterates;
};


template<class T>
double norm2_as_double(const hdnum::Vector<T>& v)
{
    double sum = 0.0;

    for (std::size_t i = 0; i < v.size(); ++i) {
        const double vi = scalar_cast<double>(v[i]);
        sum += vi * vi;
    }

    return std::sqrt(sum);
}


template<class T>
hdnum::Vector<T>
solve_with_lu_fullpivot(
    const hdnum::DenseMatrix<T>& LU,
    const hdnum::Vector<std::size_t>& p,
    const hdnum::Vector<std::size_t>& q,
    const hdnum::Vector<T>& b
)
{
    hdnum::Vector<T> y(b); // copy b 
    hdnum::Vector<T> x(b.size()); // holds solution

    hdnum::permute_forward(p, y);
    hdnum::solveL(LU, y, y); // Solve L*y = P*b, in-place in y.
    hdnum::solveR(LU, x, y); // Solve U*x = y
    hdnum::permute_backward(q, x); // undo column permutation
    return x;
}



template<class T_residual, class T_work>
hdnum::Vector<T_residual> compute_residual(
    const hdnum::DenseMatrix<T_work>& A,
    const hdnum::Vector<T_work>& b,
    const hdnum::Vector<T_work>& x
)
{
    const std::size_t n = b.size();

    hdnum::DenseMatrix<T_residual> A_r(A.rowsize(), A.colsize());
    hdnum::Vector<T_residual> b_r(n);
    hdnum::Vector<T_residual> x_r(n);
    hdnum::Vector<T_residual> Ax_r(n);
    hdnum::Vector<T_residual> r_r(n);

    convert(A_r, A);
    convert(b_r, b);
    convert(x_r, x);

    A_r.mv(Ax_r, x_r);

    for (std::size_t i = 0; i < n; i++) {
        r_r[i] = b_r[i] - Ax_r[i];
    }

    return r_r;
}


template <class T_factor, class T_work, class T_residual>
MixedIRResult<T_work> mixed_ir(
    const hdnum::DenseMatrix<T_work>& A,
    const hdnum::Vector<T_work>& b,
    const MixedIROptions& options = {}
)
{
    if (A.rowsize() != A.colsize() || A.rowsize() == 0) {
        throw std::invalid_argument("mixed_ir: A must be square and non-empty");
    }

    if (A.rowsize() != b.size()) {
        throw std::invalid_argument("mixed_ir: A must be square and nonempty");
    }

    const std::size_t n = b.size();

    const double tol = 
        options.rel_correction_tol > 0.0 ? options.rel_correction_tol
                                         : default_unit_roundoff<T_work>();

    // 1. convert A and b to factorization precision.
    hdnum::DenseMatrix<T_factor> A_f(n, n);
    hdnum::Vector<T_factor> b_f(n);

    convert(A_f, A);
    convert(b_f, b);

    // 2. Compute LU factorization in low precision
    //
    // lr_fullpivot overwrites A_f with the combined L / U factors
    hdnum::Vector<std::size_t> p(n);
    hdnum::Vector<std::size_t> q(n);

    hdnum::lr_fullpivot(A_f, p, q);


    // 3. Initial low-precision solve
    hdnum::Vector<T_factor> x0_f = solve_with_lu_fullpivot(A_f, p, q, b_f);

    // 4. cast x0 to working precision
    MixedIRResult<T_work> result;
    result.x = hdnum::Vector<T_work>(n);
    convert(result.x, x0_f);

    // append the initial solution to the iterates vector
    if (options.store_iterates) {
        result.iterates.push_back(result.x);
    }

    // 5. refinement loop
    for (std::size_t k = 0; k < options.max_iterations; k++) {
        // Compute residual r = b - A * x in residual precision.
        hdnum::Vector<T_residual> r_r = compute_residual<T_residual>(A, b, result.x);

        // cast residual to working precision
        // hdnum::Vector<T_work> r_w(n);
        // convert(r_w, r_r);

        // The triangular solve uses T_factor L and U factors,
        // so the right-hand side must also be in T_factor
        hdnum::Vector<T_factor> r_f(n);
        convert(r_f, r_r);

        // Solve correction equation A d = r using existing LU factors:
        hdnum::Vector<T_factor> d_f = solve_with_lu_fullpivot(A_f, p, q, r_f);

        // Cast correction back to working precision
        hdnum::Vector<T_work> d_w(n);
        convert(d_w, d_f);

        const double norm_d = norm2_as_double(d_w);
        const double norm_x = norm2_as_double(result.x);
        const double denominator = (norm_x > 0.0) ? norm_x : 1.0;

        const double rel_corr = norm_d / denominator;

        result.rel_corrections.push_back(rel_corr);
        result.final_rel_correction = rel_corr;

        // update in working precision
        for (std::size_t i = 0; i < n; i++) {
            result.x[i] += d_w[i];
        }

        // append the current solution to the iterates vector
        if (options.store_iterates) {
            result.iterates.push_back(result.x);
        }

        result.iterations = k + 1;

        if (norm_d <= tol * norm_x) {
            result.converged = true;
            break;
        }

    }

    return result;
}



}



