#ifndef TESTMATRICES_HH
#define TESTMATRICES_HH

#include <cmath>
#include <cstddef>
#include <random>
// #include "densematrix.hh"
// #include "vector.hh"
// #include "qr.hh"
#include "hdnum.hh"

namespace hdnum {

/*!
  \brief Generate a random orthogonal matrix via QR factorization of a random Gaussian matrix.

  The matrix Q is computed by applying modified Gram-Schmidt
  to a matrix with entries drawn from a standard normal distribution. The result
  satisfies Q^T Q = I up to working precision.

  \param[out] Q square DenseMatrix of size n x n, overwritten with an orthogonal matrix
  \param[in] seed seed for the random number generator (default: 42)
*/
template <typename REAL>
void random_orthogonal(DenseMatrix<REAL>& Q, unsigned int seed = 42)
{
    if (Q.rowsize() != Q.colsize() || Q.rowsize() == 0)
        HDNUM_ERROR("need square and nonempty matrix");
    std::size_t n = Q.rowsize();

    // fill with standard normal random entries
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Q[i][j] = REAL(dist(gen));

    // orthogonalize via modified Gram-Schmidt
    Q = modified_gram_schmidt(Q);
}

/*!
  \brief Generate a general (non-symmetric) matrix with prescribed singular values.

  Constructs A = U * diag(sigma) * V^T where U is m x n with orthonormal columns,
  V is n x n orthogonal, and sigma contains the prescribed singular values.
  The singular values are given explicitly via the vector sigma.

  \param[out] A DenseMatrix of size m x n, overwritten with the result
  \param[in] sigma Vector of length n containing the desired singular values
  \param[in] seed_u seed for generating U (default: 42)
  \param[in] seed_v seed for generating V (default: 137)
*/
template <typename REAL>
void randsvd(DenseMatrix<REAL>& A, const Vector<REAL>& sigma,
             unsigned int seed_u = 42, unsigned int seed_v = 137)
{
    std::size_t m = A.rowsize();
    std::size_t n = A.colsize();
    if (n == 0 || m == 0)
        HDNUM_ERROR("need nonempty matrix");
    if (m < n)
        HDNUM_ERROR("need m >= n (tall or square matrix)");
    if (sigma.size() != n)
        HDNUM_ERROR("sigma must have length n = A.colsize()");

    // generate random orthogonal matrices
    // for U: generate m x m orthogonal, then use first n columns
    DenseMatrix<REAL> U_full(m, m);
    random_orthogonal(U_full, seed_u);

    DenseMatrix<REAL> V(n, n);
    random_orthogonal(V, seed_v);

    // A = U(:,1:n) * diag(sigma) * V^T
    // compute column by column: A[:,j] = sum_k U[:,k] * sigma[k] * V[j,k]
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            REAL s(0.0);
            for (std::size_t k = 0; k < n; ++k)
                s += U_full[i][k] * sigma[k] * V[j][k];
            A[i][j] = s;
        }
}

/*!
  \brief Generate a general matrix with prescribed condition number and logarithmically spaced singular values.

  Constructs A = U * diag(sigma) * V^T where sigma_1 = 1, sigma_n = 1/kappa,
  and the remaining singular values are logarithmically spaced between these.

  \param[out] A DenseMatrix of size m x n, overwritten with the result
  \param[in] kappa desired condition number kappa(A) = sigma_1 / sigma_n
  \param[in] seed_u seed for generating U (default: 42)
  \param[in] seed_v seed for generating V (default: 137)
*/
template <typename REAL>
void randsvd(DenseMatrix<REAL>& A, REAL kappa,
             unsigned int seed_u = 42, unsigned int seed_v = 137)
{
    std::size_t n = A.colsize();
    if (n == 0)
        HDNUM_ERROR("need nonempty matrix");

    Vector<REAL> sigma(n);
    if (n == 1) {
        sigma[0] = REAL(1.0);
    } else {
        // logarithmically spaced singular values from 1 to 1/kappa
        REAL log_kappa = log(REAL(kappa));
        for (std::size_t i = 0; i < n; ++i)
            sigma[i] = exp(-log_kappa * REAL(i) / REAL(n - 1));
    }

    randsvd(A, sigma, seed_u, seed_v);
}

/*!
  \brief Generate a symmetric positive definite matrix with prescribed eigenvalues.

  Constructs A = Q * diag(lambda) * Q^T where Q is n x n orthogonal
  and lambda contains the prescribed eigenvalues.

  \param[out] A square DenseMatrix of size n x n, overwritten with the result
  \param[in] lambda Vector of length n containing the desired eigenvalues (must be positive)
  \param[in] seed seed for generating Q (default: 42)
*/
template <typename REAL>
void randspd(DenseMatrix<REAL>& A, const Vector<REAL>& lambda, unsigned int seed = 42)
{
    if (A.rowsize() != A.colsize() || A.rowsize() == 0)
        HDNUM_ERROR("need square and nonempty matrix");
    std::size_t n = A.rowsize();
    if (lambda.size() != n)
        HDNUM_ERROR("lambda must have length n");

    DenseMatrix<REAL> Q(n, n);
    random_orthogonal(Q, seed);

    // A = Q * diag(lambda) * Q^T
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            REAL s(0.0);
            for (std::size_t k = 0; k < n; ++k)
                s += Q[i][k] * lambda[k] * Q[j][k];
            A[i][j] = s;
            A[j][i] = s; // ensure exact symmetry
        }
}

/*!
  \brief Generate an SPD matrix with prescribed condition number and logarithmically spaced eigenvalues.

  Constructs A = Q * diag(lambda) * Q^T where lambda_1 = kappa, lambda_n = 1,
  and the remaining eigenvalues are logarithmically spaced.

  \param[out] A square DenseMatrix of size n x n, overwritten with the result
  \param[in] kappa desired condition number kappa(A) = lambda_max / lambda_min
  \param[in] seed seed for generating Q (default: 42)
*/
template <typename REAL>
void randspd(DenseMatrix<REAL>& A, REAL kappa, unsigned int seed = 42)
{
    std::size_t n = A.rowsize();
    if (n == 0)
        HDNUM_ERROR("need nonempty matrix");

    Vector<REAL> lambda(n);
    if (n == 1) {
        lambda[0] = REAL(1.0);
    } else {
        // logarithmically spaced eigenvalues from kappa down to 1
        REAL log_kappa = log(REAL(kappa));
        for (std::size_t i = 0; i < n; ++i)
            lambda[i] = exp(log_kappa * REAL(n - 1 - i) / REAL(n - 1));
    }

    randspd(A, lambda, seed);
}

/*!
  \brief Generate a Hilbert matrix.

  The Hilbert matrix has entries H(i,j) = 1 / (i + j + 1) and is
  symmetric positive definite with a condition number that grows
  rapidly with n. It is a classical test matrix for numerical algorithms.

  \param[out] A square DenseMatrix of size n x n, overwritten with the Hilbert matrix
*/
template <typename REAL>
void hilbert(DenseMatrix<REAL>& A)
{
    if (A.rowsize() != A.colsize() || A.rowsize() == 0)
        HDNUM_ERROR("need square and nonempty matrix");
    for (std::size_t i = 0; i < A.rowsize(); ++i)
        for (std::size_t j = 0; j < A.colsize(); ++j)
            A[i][j] = REAL(1.0) / REAL(i + j + 1);
}

/*!
  \brief Generate a Kahan matrix.

  The Kahan matrix is upper triangular and is specifically designed to
  challenge Gram-Schmidt orthogonalization: classical Gram-Schmidt
  loses orthogonality maximally on this matrix, while modified
  Gram-Schmidt still performs well. The condition number grows
  exponentially with n for c close to 1.

  The matrix is defined as K = diag(1, s, s^2, ..., s^{n-1}) * R
  where R is upper triangular with R(i,i) = 1 and R(i,j) = -c for j > i,
  and s = sqrt(1 - c^2).

  \param[out] A square DenseMatrix of size n x n, overwritten with the Kahan matrix
  \param[in] c parameter in (0, 1); values close to 1 give high condition numbers
*/
template <typename REAL>
void kahan(DenseMatrix<REAL>& A, REAL c)
{
    if (A.rowsize() != A.colsize() || A.rowsize() == 0)
        HDNUM_ERROR("need square and nonempty matrix");
    std::size_t n = A.rowsize();
    REAL s = sqrt(REAL(1.0) - c * c);

    // first build the upper triangular factor R
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (j < i)
                A[i][j] = REAL(0.0);
            else if (j == i)
                A[i][j] = REAL(1.0);
            else
                A[i][j] = -c;
        }

    // multiply from the left by diag(1, s, s^2, ...)
    REAL si(1.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            A[i][j] *= si;
        si *= s;
    }
}

} // namespace hdnum

#endif // TESTMATRICES_HH