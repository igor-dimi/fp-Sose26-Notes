#pragma once

#include <iostream>
#include "test_matrices.hpp"
#include "mixed_ir.hpp"
#include "hdnum_conversions.hpp"


template<class T>
void print_vector(const hdnum::Vector<T>& vec)
{
    for (size_t i = 0; i < vec.size(); i++) {
        if (i % 10 == 0) std::cout << "\n";
        std::cout << vec[i] << ", ";
    }
    std::cout << "\n";
}

int main(int argc, char const *argv[])
{    

    using T_factor = hdnum::FP16;
    using T_work = hdnum::FP64;
    using T_residual = hdnum::FP128;
    using T_measure = hdnum::FP256;

    const std::size_t n = 100;
    const double kappa = 10;

    auto problem = mpir::make_rotated_spd_problem<hdnum::FP64>(n, kappa);
    mpir::MixedIROptions options;
    options.max_iterations = 20;
    options.store_iterates = true;

    auto result = mpir::mixed_ir<T_factor, T_work, T_residual> (
        problem.A,
        problem.b,
        options
    );

    std::cout << std::boolalpha;
    std::cout << "iterations: "<< result.iterations << "\n";
    std::cout << "converged: "<< result.converged << "\n";

    // const double kappas[] = {
    //     1.0,
    //     10.0,
    //     100.0,
    //     1000.0,
    //     10000.0
    // };

    // for (double kappa : kappas) {
    //     auto problem = mpir::make_rotated_spd_problem<hdnum::FP64>(n, kappa);


    //     std::cout << "k: " << kappa << std::endl;
    //     std::cout << "A: " << problem.A << std::endl;
    //     std::cout << "b: " << problem.b << std::endl;
    // }

    // std::cout << "x0: " << "\n";
    // print_vector(result.iterates[0]);

    for (size_t i = 0; i < result.iterates.size(); i++) {
        std::cout << "x" << i << ": " << "\n";
        print_vector(result.iterates[i]);
    }

    std::cout << "result: " << "\n";
    print_vector(result.x);
    std::cout << "b: " << "\n";
    print_vector(problem.b);

    for (size_t i = 0; i < result.rel_corrections.size(); i++) {
        std::cout << "rel_corr" << i << ": " << result.rel_corrections[i] << "\n";
    }

    auto x10 = result.iterates[10];
    hdnum::Vector<T_residual> x10_r(n);
    mpir::convert(x10_r, x10);

    std::cout << "x10: " << "\n";
    print_vector(x10);

    std::cout << "x10_r: " << "\n";
    print_vector(x10_r);

    auto res10_r = mpir::compute_residual<T_residual, T_work>(problem.A, problem.b, x10);
    std::cout << "res10_r: " << "\n";
    print_vector(res10_r);

    hdnum::Vector<T_work> res10(n);

    problem.A.mv(res10, x10); //res10 == A * x10;
    res10 = problem.b - res10;


    std::cout << "res10: " << "\n";
    print_vector(res10);



    hdnum::DenseMatrix<T_residual> A_r(n, n);
    hdnum::Vector<T_residual> b_r(n);
    hdnum::Vector<T_residual> res10_r_2(n);

    mpir::convert(A_r, problem.A);
    mpir::convert(b_r, problem.b);

    A_r.mv(res10_r_2, x10_r); //res10_r_2 == A_r * x10_r
    res10_r_2 = b_r - res10_r_2;


    std::cout << "res10_r_2: " << "\n";
    print_vector(res10_r_2);

    
    


    return 0;
}



