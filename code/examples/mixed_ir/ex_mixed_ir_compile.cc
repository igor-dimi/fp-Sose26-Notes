#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "hdnum.hh"
#include "mixed_ir.hpp"   // or "mixed_ir.hpp", depending on your filename

double norm2_double(const hdnum::Vector<double>& v)
{
    double sum = 0.0;

    for (std::size_t i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }

    return std::sqrt(sum);
}

double residual_norm2(
    const hdnum::DenseMatrix<double>& A,
    const hdnum::Vector<double>& b,
    const hdnum::Vector<double>& x
)
{
    hdnum::Vector<double> Ax(b.size());
    A.mv(Ax, x);

    hdnum::Vector<double> r(b.size());

    for (std::size_t i = 0; i < b.size(); ++i) {
        r[i] = b[i] - Ax[i];
    }

    return norm2_double(r);
}

void print_result(
    const std::string& title,
    const mpir::MixedIRResult<double>& result,
    const hdnum::DenseMatrix<double>& A,
    const hdnum::Vector<double>& b
)
{
    std::cout << title << '\n';
    std::cout << std::string(title.size(), '-') << '\n';

    std::cout << "converged:              " << result.converged << '\n';
    std::cout << "iterations:             " << result.iterations << '\n';
    std::cout << "final_rel_correction:   "
              << result.final_rel_correction << '\n';

    std::cout << "solution:\n";
    for (std::size_t i = 0; i < result.x.size(); ++i) {
        std::cout << "  x[" << i << "] = " << result.x[i] << '\n';
    }

    std::cout << "residual norm ||b - A*x||_2:\n";
    std::cout << "  " << residual_norm2(A, b, result.x) << '\n';

    std::cout << "relative correction history:\n";
    for (std::size_t k = 0; k < result.rel_corrections.size(); ++k) {
        std::cout << "  step " << k + 1
                  << ": " << result.rel_corrections[k] << '\n';
    }

    std::cout << '\n';
}

int main()
{
    std::cout << std::scientific << std::setprecision(17);
    std::cout << std::boolalpha;

    hdnum::DenseMatrix<double> A(2, 2);
    A[0][0] = 2.0; A[0][1] = 1.0;
    A[1][0] = 1.0; A[1][1] = 3.0;

    hdnum::Vector<double> b(2);
    b[0] = 1.0;
    b[1] = 2.0;

    mpir::MixedIROptions options;
    options.max_iterations = 10;
    options.rel_correction_tol = 1e-14;

    auto result_double =
        mpir::mixed_ir<double, double, double>(A, b, options);

    auto result_mixed =
        mpir::mixed_ir<hdnum::FP8, double, hdnum::FP128>(A, b, options);

    print_result("All-double version", result_double, A, b);

    print_result(
        "Mixed version: factorization FP16, work double, residual FP128",
        result_mixed,
        A,
        b
    );

    std::cout << "Componentwise comparison\n";
    std::cout << "------------------------\n";

    for (std::size_t i = 0; i < result_double.x.size(); ++i) {
        std::cout << "component " << i << '\n';
        std::cout << "  all-double: " << result_double.x[i] << '\n';
        std::cout << "  mixed:      " << result_mixed.x[i] << '\n';
        std::cout << "  difference: "
                  << result_mixed.x[i] - result_double.x[i] << '\n';
    }

    return 0;
}