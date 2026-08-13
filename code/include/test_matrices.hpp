#pragma once

/**
 * @file test_matrices.hpp
 * @brief Construction of reproducible linear systems for numerical experiments.
 *
 * This header provides a common interface for generating the structured
 * rotated-SPD, random-SPD, and random-SVD test problems used by the project.
 * Matrices and right-hand sides are stored in T_data, while reference solutions
 * are computed by solving the stored system in T_reference. Right-hand-side
 * construction and random seeds are controlled through TestProblemOptions.
 */

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


/**
 * @brief Available constructions of the right-hand side.
 */
enum class RightHandSideMode {
    ones_solution,        ///< Form b = A*x with x = (1, ..., 1).
    random_sign_solution, ///< Form b = A*x with reproducible entries x_i in {-1, 1}.
    random_normal_rhs     ///< Draw the entries of b from the standard normal distribution.
};


/**
 * @brief Configuration for reproducible test-problem generation.
 */
struct TestProblemOptions {
    /// Method used to construct the right-hand side.
    RightHandSideMode rhs_mode =
        RightHandSideMode::ones_solution;

    /// Seed for the first random orthogonal factor.
    unsigned int matrix_seed_u = 42;

    /// Seed for the second random orthogonal factor of random-SVD matrices.
    unsigned int matrix_seed_v = 137;

    /// Seed for a random solution or right-hand side.
    unsigned int vector_seed = 2718;

    /// Rotation angle, in radians, for the structured rotated-SPD matrix.
    double rotation_theta = 0.3;
};


/**
 * @brief Complete linear system and its reference solution.
 *
 * @tparam T_data Storage precision of the matrix and right-hand side.
 * @tparam T_reference Arithmetic and storage precision of the reference
 *         solution.
 */
template<class T_data, class T_reference = T_data>
struct LinearSystem {
    /// Coefficient matrix in data precision.
    hdnum::DenseMatrix<T_data> A;

    /// Right-hand side in data precision.
    hdnum::Vector<T_data> b;

    /// Reference solution of the stored system A*x = b.
    hdnum::Vector<T_reference> x_true;

    /// Requested condition number; not recomputed after storage rounding.
    double kappa;
};


/**
 * @brief Completes a test problem after its coefficient matrix is generated.
 *
 * Depending on TestProblemOptions::rhs_mode, this function forms b from an
 * all-ones or random-sign solution, or draws b directly from a standard normal
 * distribution. It then recomputes the solution of the stored system in
 * T_reference, thereby accounting for rounding in A and b.
 *
 * @tparam T_data Storage precision of the matrix and right-hand side.
 * @tparam T_reference Arithmetic and result precision of the reference solve.
 * @param A Nonempty square coefficient matrix. The matrix is moved into the
 *        returned system.
 * @param kappa Requested condition number stored as problem metadata.
 * @param options Right-hand-side mode and random seeds.
 * @return Complete test system containing A, b, the reference solution, and
 *         the requested condition number.
 *
 * @throws std::invalid_argument If the reference solve rejects the stored
 *         system as empty, nonsquare, or dimensionally incompatible.
 */
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


/**
 * @brief Constructs a structured symmetric positive-definite test problem.
 *
 * The eigenvalues are logarithmically spaced from kappa^(-1/2) to
 * kappa^(1/2). Adjacent pairs are mixed by equal planar rotations, producing a
 * block-diagonal SPD matrix with the requested spectral condition number.
 *
 * @tparam T_data Storage precision of the matrix and right-hand side.
 * @tparam T_reference Arithmetic and result precision of the reference solve.
 * @param n Matrix dimension.
 * @param kappa Requested spectral condition number.
 * @param options Right-hand-side settings and rotation angle.
 * @return Complete structured rotated-SPD test system.
 *
 * @pre kappa > 0.
 * @throws std::invalid_argument If n is zero during the reference solve.
 */
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

            // Center the spectrum geometrically around one.
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

            // Form R*diag(a,d)*R^T for the current 2-by-2 block.
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


/**
 * @brief Constructs a dense random symmetric positive-definite test problem.
 *
 * The matrix is generated by hdnum::randspd with logarithmically spaced
 * eigenvalues and the requested spectral condition number.
 *
 * @tparam T_data Storage precision of the matrix and right-hand side.
 * @tparam T_reference Arithmetic and result precision of the reference solve.
 * @param n Matrix dimension.
 * @param kappa Requested spectral condition number.
 * @param options Right-hand-side settings and matrix seed.
 * @return Complete random-SPD test system.
 *
 * @pre n > 0 and kappa > 0.
 * @note Invalid generator inputs are reported by hdnum::randspd.
 */
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


/**
 * @brief Constructs a dense random general test problem by SVD.
 *
 * The matrix is generated by hdnum::randsvd with logarithmically spaced
 * singular values and the requested 2-norm condition number.
 *
 * @tparam T_data Storage precision of the matrix and right-hand side.
 * @tparam T_reference Arithmetic and result precision of the reference solve.
 * @param n Matrix dimension.
 * @param kappa Requested 2-norm condition number.
 * @param options Right-hand-side settings and seeds for both orthogonal factors.
 * @return Complete random-SVD test system.
 *
 * @pre n > 0 and kappa > 0.
 * @note Invalid generator inputs are reported by hdnum::randsvd.
 */
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
