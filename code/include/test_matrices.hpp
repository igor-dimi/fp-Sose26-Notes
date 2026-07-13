#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"
#include "reference_solve.hpp"
#include "testmatrix.hh"

namespace mpir {


enum class RightHandSideMode {
    ones_solution,
    random_sign_solution,
    random_normal_rhs
};


struct TestProblemOptions {
    RightHandSideMode rhs_mode =
        RightHandSideMode::ones_solution;

    unsigned int matrix_seed_u = 42;
    unsigned int matrix_seed_v = 137;
    unsigned int vector_seed = 2718;

    double rotation_theta = 0.3;
};


template<class T_data, class T_reference = T_data>
struct LinearSystem {
    hdnum::DenseMatrix<T_data> A;
    hdnum::Vector<T_data> b;
    hdnum::Vector<T_reference> x_true;
    double kappa;
};


// Complete a problem after its matrix A has been generated.
//
// Depending on options.rhs_mode, this function either:
//
//   1. constructs x = (1,...,1) and forms b = A*x,
//   2. constructs a random-sign x and forms b = A*x, or
//   3. constructs a normally distributed random b.
//
// In every case, x_true is computed by solving the stored system
// A*x = b in T_reference.
template<class T_data, class T_reference>
LinearSystem<T_data, T_reference>
complete_problem(
    hdnum::DenseMatrix<T_data> A,
    double kappa,
    const TestProblemOptions& options)
{
    const std::size_t n = A.rowsize();

    hdnum::Vector<T_data> b(n);

    if (options.rhs_mode ==
        RightHandSideMode::ones_solution) {

        hdnum::Vector<T_data> x_constructed(n);

        for (std::size_t i = 0; i < n; ++i) {
            x_constructed[i] = T_data(1);
        }

        A.mv(b, x_constructed);
    }
    else if (options.rhs_mode ==
             RightHandSideMode::random_sign_solution) {

        std::mt19937 gen(options.vector_seed);
        std::bernoulli_distribution positive(0.5);

        hdnum::Vector<T_data> x_constructed(n);

        for (std::size_t i = 0; i < n; ++i) {
            x_constructed[i] =
                positive(gen) ? T_data(1) : T_data(-1);
        }

        A.mv(b, x_constructed);
    }
    else {
        std::mt19937 gen(options.vector_seed);
        std::normal_distribution<double> normal(0.0, 1.0);

        for (std::size_t i = 0; i < n; ++i) {
            b[i] = scalar_cast<T_data>(normal(gen));
        }
    }

    auto x_reference =
        high_precision_solve<T_reference, T_data>(A, b);

    return LinearSystem<T_data, T_reference>{
        std::move(A),
        std::move(b),
        std::move(x_reference),
        kappa
    };
}


// Structured block-rotated SPD problem.
template<class T_data, class T_reference = T_data>
LinearSystem<T_data, T_reference>
make_rotated_spd_problem(
    std::size_t n,
    double kappa,
    const TestProblemOptions& options = {})
{
    hdnum::DenseMatrix<T_data> A(n, n, T_data(0));

    std::vector<double> lambda(n);

    if (n == 1) {
        lambda[0] = 1.0;
    }
    else {
        for (std::size_t i = 0; i < n; ++i) {
            const double alpha =
                static_cast<double>(i) /
                static_cast<double>(n - 1);

            // Eigenvalues from kappa^(-1/2)
            // to kappa^(+1/2).
            lambda[i] =
                std::pow(kappa, alpha - 0.5);
        }
    }

    const double theta = options.rotation_theta;
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    for (std::size_t i = 0; i < n; i += 2) {
        if (i + 1 < n) {
            const double a = lambda[i];
            const double d = lambda[i + 1];

            const double A00 =
                c * c * a + s * s * d;

            const double A11 =
                s * s * a + c * c * d;

            const double A01 =
                c * s * (a - d);

            A[i][i] =
                scalar_cast<T_data>(A00);

            A[i + 1][i + 1] =
                scalar_cast<T_data>(A11);

            A[i][i + 1] =
                scalar_cast<T_data>(A01);

            A[i + 1][i] =
                scalar_cast<T_data>(A01);
        }
        else {
            A[i][i] =
                scalar_cast<T_data>(lambda[i]);
        }
    }

    return complete_problem<T_data, T_reference>(
        std::move(A),
        kappa,
        options
    );
}


// Dense random SPD problem generated with the supervisor's
// randspd implementation.
template<class T_data, class T_reference = T_data>
LinearSystem<T_data, T_reference>
make_random_spd_problem(
    std::size_t n,
    double kappa,
    const TestProblemOptions& options = {})
{
    hdnum::DenseMatrix<T_data> A(n, n);

    hdnum::randspd(
        A,
        scalar_cast<T_data>(kappa),
        options.matrix_seed_u
    );

    return complete_problem<T_data, T_reference>(
        std::move(A),
        kappa,
        options
    );
}


// Dense random general matrix generated with the supervisor's
// randsvd implementation.
template<class T_data, class T_reference = T_data>
LinearSystem<T_data, T_reference>
make_random_svd_problem(
    std::size_t n,
    double kappa,
    const TestProblemOptions& options = {})
{
    hdnum::DenseMatrix<T_data> A(n, n);

    hdnum::randsvd(
        A,
        scalar_cast<T_data>(kappa),
        options.matrix_seed_u,
        options.matrix_seed_v
    );

    return complete_problem<T_data, T_reference>(
        std::move(A),
        kappa,
        options
    );
}


} // namespace mpir