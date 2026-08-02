#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

#include "hdnum.hh"
#include "mixed_ir.hpp"
#include "test_matrices.hpp"

int main()
{
    using T_factor = hdnum::FP16;
    using T_work = hdnum::FP64;
    using T_residual = hdnum::FP128;
    using T_measure = hdnum::FP256;

    std::cout << "hey there" << std::endl;

    const std::filesystem::path output_dir = MPIR_RESULTS_RAW_DIR;
    std::filesystem::create_directories(output_dir);

    const std::filesystem::path output_file =
        output_dir /
        "residual_conversion_unscaled_fp16_fp64_fp128.csv";

    std::ofstream out(output_file);

    if (!out) {
        std::cerr << "Error: could not open output file: "
                  << output_file << "\n";
        return 1;
    }

    // Preserve every significant digit of the double-valued diagnostic
    // fields when writing them to CSV.
    out << std::scientific
        << std::setprecision(
               std::numeric_limits<double>::max_digits10
           );

    const std::size_t n = 100;
    const double kappa = 10.0;

    mpir::TestProblemOptions problem_options;
    problem_options.rhs_mode =
        mpir::RightHandSideMode::random_normal_rhs;

    auto problem =
        mpir::make_random_spd_problem<T_work, T_measure>(
            n,
            kappa,
            problem_options
        );

    mpir::MixedIROptions<T_work> mir_options;
    mir_options.max_iterations = 20;
    mir_options.store_iterates = false;
    mir_options.detect_divergence = true;
    mir_options.record_residual_diagnostics = true;

    // This is the unscaled baseline: mixed_ir converts r_k directly
    // from residual precision to factorization precision.
    auto result =
        mpir::mixed_ir<T_factor, T_work, T_residual>(
            problem.A,
            problem.b,
            mir_options
        );

    out << "iteration,residual_inf_norm,min_nonzero_abs,"
           "nonzero_components,zeroed_by_conversion\n";

    for (const auto& diagnostic :
         result.residual_diagnostics) {
        out << diagnostic.iteration << ","
            << diagnostic.residual_inf_norm << ","
            << diagnostic.min_nonzero_abs << ","
            << diagnostic.nonzero_components << ","
            << diagnostic.zeroed_by_conversion << "\n";
    }

    if (!out) {
        std::cerr << "Error: failed while writing output file: "
                  << output_file << "\n";
        return 1;
    }

    std::cout << "Writing results to: " << output_file << "\n"
              << "kappa = " << kappa
              << ", converged = " << result.converged()
              << ", iterations = " << result.iterations
              << ", final_rel_correction = "
              << result.final_rel_correction
              << ", diagnostic_records = "
              << result.residual_diagnostics.size()
              << "\n";

    return 0;
}
