#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "condition_grids.hpp"
#include "error_metrics.hpp"
#include "experiment_io.hpp"
#include "hdnum.hh"
#include "mixed_ir.hpp"
#include "test_matrices.hpp"

namespace {

constexpr std::size_t problem_dimension = 100;


/**
 * @brief Runs the convergence-history experiment for one precision
 *        configuration.
 *
 * A separate CSV file is produced for each configuration. Every requested
 * condition number contributes one row per available iterate. If the
 * algorithm fails before constructing the initial iterate, one row with empty
 * iteration-specific fields is written so that the failed run remains visible
 * in the dataset.
 *
 * @tparam T_factor   Precision used for factorization and correction solves.
 * @tparam T_work     Precision used to store and update the iterate.
 * @tparam T_residual Precision used to compute the residual.
 * @tparam T_measure  Precision used to evaluate the reported errors.
 *
 * @param precision_names Human-readable names used in CSV metadata and the
 *                        generated filename.
 * @param kappas          Requested condition numbers.
 */
template<
    class T_factor,
    class T_work,
    class T_residual,
    class T_measure
>
void run_convergence_histories(
    const mpir::PrecisionNames& precision_names,
    const std::vector<double>& kappas)
{
    mpir::TestProblemOptions problem_options;
    problem_options.rhs_mode =
        mpir::RightHandSideMode::random_normal_rhs;

    const mpir::ExperimentDescription experiment{
        mpir::ExperimentKind::convergence_history,
        mpir::MatrixFamily::random_spd,
        problem_dimension,
        precision_names
    };

    const auto output_directory =
        mpir::make_output_directory(
            MPIR_RESULTS_RAW_DIR,
            experiment.kind
        );

    const auto output_file =
        output_directory /
        mpir::make_experiment_filename(
            experiment,
            problem_options
        );

    std::ofstream out(output_file);

    if (!out) {
        throw std::runtime_error(
            "Could not open output file: " + output_file.string()
        );
    }

    out << std::scientific
        << std::setprecision(
               std::numeric_limits<double>::max_digits10
           );

    mpir::write_common_csv_header(out);
    out << ",iteration,forward_error_inf,backward_error_inf,"
           "rel_correction\n";

    mpir::MixedIROptions<T_work> algorithm_options;
    algorithm_options.max_iterations = 20;
    algorithm_options.store_iterates = true;
    algorithm_options.detect_divergence = true;
    algorithm_options.scale_residual = true;
    algorithm_options.record_residual_diagnostics = false;

    constexpr std::string_view variant = "scaled";

    for (const double kappa : kappas) {
        // The generator is reinitialized with the same fixed seeds for every
        // kappa and precision configuration. Only the prescribed spectrum
        // changes between problems.
        const auto problem =
            mpir::make_random_spd_problem<T_work, T_measure>(
                problem_dimension,
                kappa,
                problem_options
            );

        const auto result =
            mpir::mixed_ir<T_factor, T_work, T_residual>(
                problem.A,
                problem.b,
                algorithm_options
            );

        if (!result.iterates.empty()
            && result.iterates.size() != result.iterations + 1) {
            throw std::logic_error(
                "Stored-iterate count is inconsistent with completed updates"
            );
        }

        if (result.rel_corrections.size() != result.iterations) {
            throw std::logic_error(
                "Relative-correction count is inconsistent with completed updates"
            );
        }

        if (result.iterates.empty()) {
            mpir::write_common_csv_fields(
                out,
                experiment,
                problem_options,
                algorithm_options,
                kappa,
                variant,
                result
            );

            // No initial iterate exists, so all history-specific fields are
            // intentionally empty.
            out << ",,,,\n";
        }
        else {
            for (std::size_t iteration = 0;
                 iteration < result.iterates.size();
                 ++iteration) {
                const double forward_error =
                    mpir::relative_forward_error_inf<T_measure>(
                        result.iterates[iteration],
                        problem.x_true
                    );

                const double backward_error =
                    mpir::normwise_backward_error_inf<T_measure>(
                        problem.A,
                        problem.b,
                        result.iterates[iteration]
                    );

                mpir::write_common_csv_fields(
                    out,
                    experiment,
                    problem_options,
                    algorithm_options,
                    kappa,
                    variant,
                    result
                );

                out << ',' << iteration
                    << ',' << forward_error
                    << ',' << backward_error
                    << ',';

                // rel_corrections[k] describes the update from x_k to
                // x_{k+1}. Therefore, the value associated with x_i is stored
                // at index i - 1. No correction is associated with x_0.
                if (iteration > 0) {
                    out << result.rel_corrections[iteration - 1];
                }

                out << '\n';
            }
        }

        std::cout
            << precision_names.factor << '-'
            << precision_names.work << '-'
            << precision_names.residual
            << ": kappa = " << kappa
            << ", status = " << mpir::to_string(result.status)
            << ", iterations = " << result.iterations
            << ", final_rel_correction = "
            << result.final_rel_correction
            << '\n';
    }

    if (!out) {
        throw std::runtime_error(
            "Failed while writing output file: " + output_file.string()
        );
    }

    std::cout << "Writing results to: " << output_file << "\n";
}

} // namespace


int main()
{
    try {
        using FP8 = hdnum::CPFloat<4, 4>;
        using BFloat16 = hdnum::CPFloat<8, 8>;

        const std::vector<double> fp64_baseline_kappas{
            1.0,
            1.0e4,
            1.0e8,
            1.0e12
        };

        run_convergence_histories<
            hdnum::FP64,
            hdnum::FP64,
            hdnum::FP64,
            hdnum::FP256
        >(
            {"fp64", "fp64", "fp64", "fp256"},
            fp64_baseline_kappas
        );

        run_convergence_histories<
            hdnum::FP32,
            hdnum::FP64,
            hdnum::FP64,
            hdnum::FP256
        >(
            {"fp32", "fp64", "fp64", "fp256"},
            mpir::representative_kappas<hdnum::FP32>()
        );

        run_convergence_histories<
            hdnum::FP32,
            hdnum::FP64,
            hdnum::FP128,
            hdnum::FP256
        >(
            {"fp32", "fp64", "fp128", "fp256"},
            mpir::representative_kappas<hdnum::FP32>()
        );

        run_convergence_histories<
            hdnum::FP16,
            hdnum::FP64,
            hdnum::FP64,
            hdnum::FP256
        >(
            {"fp16", "fp64", "fp64", "fp256"},
            mpir::representative_kappas<hdnum::FP16>()
        );

        run_convergence_histories<
            hdnum::FP16,
            hdnum::FP64,
            hdnum::FP128,
            hdnum::FP256
        >(
            {"fp16", "fp64", "fp128", "fp256"},
            mpir::representative_kappas<hdnum::FP16>()
        );

        run_convergence_histories<
            hdnum::bfloat16,
            hdnum::FP64,
            hdnum::FP128,
            hdnum::FP256
        >(
            {"bfloat16", "fp64", "fp128", "fp256"},
            mpir::representative_kappas<hdnum::bfloat16>()
        );

        // Optional stress-test configuration from the experiment plan.
        run_convergence_histories<
            FP8,
            hdnum::FP64,
            hdnum::FP128,
            hdnum::FP256
        >(
            {"fp8", "fp64", "fp128", "fp256"},
            mpir::representative_kappas<FP8>()
        );
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
