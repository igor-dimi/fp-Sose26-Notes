#include <filesystem>
#include <fstream>
#include <iostream>

#include "hdnum.hh"
#include "mixed_ir.hpp"
#include "test_matrices.hpp"
#include "error_metrics.hpp"

int main()
{
    using T_factor = hdnum::FP64;
    using T_work = hdnum::FP64;
    using T_residual = hdnum::FP128;
    using T_measure = hdnum::FP256;

    const std::filesystem::path output_dir = MPIR_RESULTS_RAW_DIR;
    std::filesystem::create_directories(output_dir);

    const std::filesystem::path output_file =
        output_dir / "convergence_random_spd_fp16_fp64_fp128.csv";

    std::ofstream out(output_file);

    if (!out) {
        std::cerr << "Error: could not open output file: "
                << output_file << "\n";
        return 1;
    }

    std::cout << "Writing results to: " << output_file << "\n";


    const std::size_t n = 100;

    // const double kappas[] = {
    //     1.0,
    //     10.0,
    //     100.0,
    //     1000.0,
    //     10000.0
    // };

    std::vector<double> kappas;
    // double kappa = 1.0;
    // kappas.push_back(kappa);
    // for (int i = 0; i < 12; i++) {
    //     kappa *= 10.0;
    //     kappas.push_back(kappa);
    // }

    // kappas.push_back(1e5);
    kappas.push_back(1e4);
    kappas.push_back(1e1);
    kappas.push_back(1e2);
    kappas.push_back(1e6);
    kappas.push_back(1e8);
    kappas.push_back(1e10);
    kappas.push_back(1e11);
    kappas.push_back(1e9);






    out << "kappa,iteration,forward_error_inf,backward_error_inf,rel_correction,"
       "converged,total_iterations,final_rel_correction\n";

    for (double kappa : kappas) {
        mpir::TestProblemOptions problem_options;

        problem_options.rhs_mode = mpir::RightHandSideMode::random_normal_rhs;
        auto problem = mpir::make_random_spd_problem<T_work, T_measure>(n, 
            kappa,
            problem_options);

        mpir::MixedIROptions<T_work> options;
        options.max_iterations = 20;
        options.store_iterates = true;

        auto result =
            mpir::mixed_ir<T_factor, T_work, T_residual>(
                problem.A,
                problem.b,
                options
            );

        std::cout << "kappa = " << kappa
          << ", converged = " << result.converged
          << ", iterations = " << result.iterations
          << ", final_rel_correction = " << result.final_rel_correction
          << "\n";

        for (std::size_t i = 0; i < result.iterates.size(); ++i) {
            const double ferr =
                mpir::relative_forward_error_inf<T_measure>(
                    result.iterates[i],
                    problem.x_true
                );

            const double berr =
                mpir::normwise_backward_error_inf<T_measure>(
                    problem.A,
                    problem.b,
                    result.iterates[i]
                );

            double rel_corr = 0.0;

            // rel_corrections[j] corresponds to correction from x_j to x_{j+1}.
            if (i > 0 && i - 1 < result.rel_corrections.size()) {
                rel_corr = result.rel_corrections[i - 1];
            }
            out << kappa << ","
                << i << ","
                << ferr << ","
                << berr << ","
                << rel_corr << ","
                << result.converged << ","
                << result.iterations << ","
                << result.final_rel_correction
                << "\n";
        }
    }


    return 0;
}