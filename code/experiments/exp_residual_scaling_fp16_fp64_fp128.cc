#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "error_metrics.hpp"
#include "experiment_io.hpp"
#include "hdnum.hh"
#include "mixed_ir.hpp"
#include "test_matrices.hpp"

namespace {

constexpr std::size_t problem_dimension = 100;
constexpr double requested_kappa = 500.0;


template<class T_work>
void validate_histories(const mpir::MixedIRResult<T_work>& result)
{
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
}


template<class T_work>
void write_diagnostics(
    std::ostream& out,
    const mpir::ExperimentDescription& experiment,
    const mpir::TestProblemOptions& problem_options,
    const mpir::MixedIROptions<T_work>& algorithm_options,
    std::string_view variant,
    const mpir::MixedIRResult<T_work>& result)
{
    for (const auto& diagnostic : result.residual_diagnostics) {
        mpir::write_common_csv_fields(
            out,
            experiment,
            problem_options,
            algorithm_options,
            requested_kappa,
            variant,
            result
        );

        out << ',' << diagnostic.iteration
            << ',' << diagnostic.residual_inf_norm
            << ',' << diagnostic.min_nonzero_abs
            << ',' << diagnostic.nonzero_components
            << ',' << diagnostic.zeroed_by_conversion
            << ',';

        // rel_corrections[k] describes the attempted update from x_k to
        // x_{k+1}. It is absent if that update was not completed.
        if (diagnostic.iteration < result.rel_corrections.size()) {
            out << result.rel_corrections[diagnostic.iteration];
        }

        out << '\n';
    }
}


template<class T_work, class T_measure, class Problem>
void write_error_history(
    std::ostream& out,
    const mpir::ExperimentDescription& experiment,
    const mpir::TestProblemOptions& problem_options,
    const mpir::MixedIROptions<T_work>& algorithm_options,
    std::string_view variant,
    const Problem& problem,
    const mpir::MixedIRResult<T_work>& result)
{
    if (result.iterates.empty()) {
        mpir::write_common_csv_fields(
            out,
            experiment,
            problem_options,
            algorithm_options,
            requested_kappa,
            variant,
            result
        );

        // No initial iterate is available, so the history fields are empty.
        out << ",,,,\n";
        return;
    }

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
            requested_kappa,
            variant,
            result
        );

        out << ',' << iteration
            << ',' << forward_error
            << ',' << backward_error
            << ',';

        // rel_corrections[k] produces x_{k+1}, so the correction associated
        // with x_i is stored at i - 1. No correction is associated with x_0.
        if (iteration > 0) {
            out << result.rel_corrections[iteration - 1];
        }

        out << '\n';
    }
}

} // namespace


int main()
{
    try {
        using T_factor = hdnum::FP16;
        using T_work = hdnum::FP64;
        using T_residual = hdnum::FP128;
        using T_measure = hdnum::FP256;

        mpir::TestProblemOptions problem_options;
        problem_options.rhs_mode =
            mpir::RightHandSideMode::random_normal_rhs;

        const mpir::ExperimentDescription experiment{
            mpir::ExperimentKind::residual_scaling,
            mpir::MatrixFamily::random_spd,
            problem_dimension,
            {"fp16", "fp64", "fp128", "fp256"}
        };

        const auto output_directory =
            mpir::make_output_directory(
                MPIR_RESULTS_RAW_DIR,
                experiment.kind
            );

        const auto diagnostics_file =
            output_directory /
            mpir::make_experiment_filename(
                experiment,
                problem_options,
                "comparison"
            );

        // Keep error histories in a separate raw-data directory. This also
        // prevents the existing diagnostic plotter from treating them as
        // residual-diagnostic CSV files during its default file discovery.
        const auto history_directory =
            output_directory / "error_histories";
        std::filesystem::create_directories(history_directory);

        const auto history_file =
            history_directory /
            mpir::make_experiment_filename(
                experiment,
                problem_options,
                "error-history"
            );

        std::ofstream diagnostics_out(diagnostics_file);
        std::ofstream history_out(history_file);

        if (!diagnostics_out) {
            throw std::runtime_error(
                "Could not open output file: "
                + diagnostics_file.string()
            );
        }

        if (!history_out) {
            throw std::runtime_error(
                "Could not open output file: "
                + history_file.string()
            );
        }

        const auto configure_output = [](std::ostream& out) {
            out << std::scientific
                << std::setprecision(
                       std::numeric_limits<double>::max_digits10
                   );
        };

        configure_output(diagnostics_out);
        configure_output(history_out);

        mpir::write_common_csv_header(diagnostics_out);
        diagnostics_out
            << ",iteration,residual_inf_norm,min_nonzero_abs,"
               "nonzero_components,zeroed_by_conversion,rel_correction\n";

        mpir::write_common_csv_header(history_out);
        history_out
            << ",iteration,forward_error_inf,backward_error_inf,"
               "rel_correction\n";

        // Both variants use this exact matrix, right-hand side, and reference
        // solution. Only residual scaling differs between the runs.
        const auto problem =
            mpir::make_random_spd_problem<T_work, T_measure>(
                problem_dimension,
                requested_kappa,
                problem_options
            );

        const auto run =
            [&](std::string_view variant, bool scale_residual) {
                mpir::MixedIROptions<T_work> algorithm_options;
                algorithm_options.max_iterations = 20;
                algorithm_options.store_iterates = true;
                algorithm_options.detect_divergence = true;
                algorithm_options.scale_residual = scale_residual;
                algorithm_options.record_residual_diagnostics = true;

                const auto result =
                    mpir::mixed_ir<T_factor, T_work, T_residual>(
                        problem.A,
                        problem.b,
                        algorithm_options
                    );

                validate_histories(result);

                write_diagnostics(
                    diagnostics_out,
                    experiment,
                    problem_options,
                    algorithm_options,
                    variant,
                    result
                );

                write_error_history<T_work, T_measure>(
                    history_out,
                    experiment,
                    problem_options,
                    algorithm_options,
                    variant,
                    problem,
                    result
                );

                std::cout
                    << variant
                    << ": status = " << mpir::to_string(result.status)
                    << ", iterations = " << result.iterations
                    << ", final_rel_correction = "
                    << result.final_rel_correction
                    << ", diagnostic_records = "
                    << result.residual_diagnostics.size()
                    << ", stored_iterates = "
                    << result.iterates.size()
                    << '\n';
            };

        run("unscaled", false);
        run("scaled", true);

        if (!diagnostics_out) {
            throw std::runtime_error(
                "Failed while writing output file: "
                + diagnostics_file.string()
            );
        }

        if (!history_out) {
            throw std::runtime_error(
                "Failed while writing output file: "
                + history_file.string()
            );
        }

        std::cout
            << "Writing diagnostics to: " << diagnostics_file << '\n'
            << "Writing error histories to: " << history_file << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
