#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "hdnum.hh"
#include "hdnum_conversions.hpp"

namespace mpir {

template<class T>
struct LinearSystem {
    hdnum::DenseMatrix<T> A;
    hdnum::Vector<T> b;
    hdnum::Vector<T> x_true;
    double kappa;
};

template<class T>
LinearSystem<T>
make_rotated_spd_problem(std::size_t n, double kappa, double theta = 0.3)
{
    hdnum::DenseMatrix<T> A(n, n, T(0));
    hdnum::Vector<T> x_true(n);
    hdnum::Vector<T> b(n);

    for (std::size_t i = 0; i < n; ++i) {
        x_true[i] = T(1);
    }

    std::vector<double> lambda(n);

    if (n == 1) {
        lambda[0] = 1.0;
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            const double alpha = static_cast<double>(i) / static_cast<double>(n - 1);

            // Eigenvalues from kappa^(-1/2) to kappa^(+1/2).
            lambda[i] = std::pow(kappa, alpha - 0.5);
        }
    }

    const double c = std::cos(theta);
    const double s = std::sin(theta);

    for (std::size_t i = 0; i < n; i += 2) {
        if (i + 1 < n) {
            const double a = lambda[i];
            const double d = lambda[i + 1];

            const double A00 = c * c * a + s * s * d;
            const double A11 = s * s * a + c * c * d;
            const double A01 = c * s * (a - d);

            A[i][i]         = scalar_cast<T>(A00);
            A[i + 1][i + 1] = scalar_cast<T>(A11);
            A[i][i + 1]     = scalar_cast<T>(A01);
            A[i + 1][i]     = scalar_cast<T>(A01);
        } else {
            A[i][i] = scalar_cast<T>(lambda[i]);
        }
    }

    A.mv(b, x_true);

    return LinearSystem<T>{A, b, x_true, kappa};
}

} // namespace mpir