#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

#include "experiment_io.hpp"
#include "hdnum.hh"
#include "mixed_ir.hpp"
#include "test_matrices.hpp"

int main()
{
    using T_factor = hdnum::FP16;
    using T_work = hdnum::FP64;
    using T_residual = hdnum::FP128;
    using T_measure = hdnum::FP256;

    const std::size_t n = 100;
    const double kappa = 10.0;

    mpir::TestProblemOptions problem_options;
    problem_options.rhs_mode =
        mpir::RightHandSideMode::random_normal_rhs;

    const mpir::ExperimentDescription experiment{
        mpir::ExperimentKind::residual_scaling,
        mpir::MatrixFamily::random_spd,
        n,
        {"fp16", "fp64", "fp128", "fp256"}
    };

    const auto output_dir =
        mpir::make_output_directory(
            MPIR_RESULTS_RAW_DIR,
            experiment.kind
        );

    const auto output_file =
        output_dir /
        mpir::make_experiment_filename(
            experiment,
            problem_options,
            "comparison"
        );

    std::ofstream out(output_file);

    if (!out) {
        std::cerr << "Error: could not open output file: "
                  << output_file << "\n";
        return 1;
    }

    out << std::scientific
        << std::setprecision(
               std::numeric_limits<double>::max_digits10
           );

    // Both runs use this same matrix and right-hand side, so any
    // difference is caused by residual scaling rather than a new
    // random problem instance.
    const auto problem =
        mpir::make_random_spd_problem<T_work, T_measure>(
            n,
            kappa,
            problem_options
        );

    mpir::write_common_csv_header(out);
    out << ",iteration,residual_inf_norm,min_nonzero_abs,"
           "nonzero_components,zeroed_by_conversion,rel_correction\n";

    const auto run =
        [&](std::string_view variant, bool scale_residual) {
            mpir::MixedIROptions<T_work> options;
            options.max_iterations = 20;
            options.store_iterates = false;
            options.detect_divergence = true;
            options.scale_residual = scale_residual;
            options.record_residual_diagnostics = true;

            const auto result =
                mpir::mixed_ir<T_factor, T_work, T_residual>(
                    problem.A,
                    problem.b,
                    options
                );

            for (const auto& diagnostic :
                 result.residual_diagnostics) {
                const double rel_correction =
                    diagnostic.iteration <
                            result.rel_corrections.size()
                        ? result.rel_corrections[
                              diagnostic.iteration
                          ]
                        : std::numeric_limits<double>::quiet_NaN();

                mpir::write_common_csv_fields(
                    out,
                    experiment,
                    problem_options,
                    options,
                    kappa,
                    variant,
                    result
                );

                out << "," << diagnostic.iteration << ","
                    << diagnostic.residual_inf_norm << ","
                    << diagnostic.min_nonzero_abs << ","
                    << diagnostic.nonzero_components << ","
                    << diagnostic.zeroed_by_conversion << ","
                    << rel_correction << "\n";
            }

            std::cout << variant
                      << ": converged = " << result.converged()
                      << ", iterations = " << result.iterations
                      << ", final_rel_correction = "
                      << result.final_rel_correction
                      << ", diagnostic_records = "
                      << result.residual_diagnostics.size()
                      << "\n";
        };

    run("unscaled", false);
    run("scaled", true);

    if (!out) {
        std::cerr << "Error: failed while writing output file: "
                  << output_file << "\n";
        return 1;
    }

    std::cout << "Writing results to: " << output_file << "\n";
    return 0;
}
