#pragma once

#include <iostream>
#include "test_matrices.hpp"


int main(int argc, char const *argv[])
{
    const std::size_t n = 6;

    const double kappas[] = {
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0
    };

    for (double kappa : kappas) {
        auto problem = mpir::make_rotated_spd_problem<hdnum::FP64>(n, kappa);


        std::cout << "k: " << kappa << std::endl;
        std::cout << "A: " << problem.A << std::endl;
        std::cout << "b: " << problem.b << std::endl;
    }

    return 0;
}



