#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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
constexpr double minimum_kappa = 1.0;
constexpr double max_boundary_multiple = 10.0;
constexpr std::size_t points_per_decade = 10;


/**
 * @brief Returns the condition-number grid used by Group C.
 *
 * Both residual-precision configurations use the same FP32 factorization
 * precision. The common grid therefore begins at kappa = 1, uses ten
 * logarithmic subdivisions per decade, extends to 10/u_f, and contains the
 * exact theoretical boundary 1/u_f.
 */
[[nodiscard]] std::vector<double> group_c_kappas()
{
    return mpir::kappa_sweep<hdnum::FP32>(
        minimum_kappa,
        max_boundary_multiple,
        points_per_decade
    );
}


/**
 * @brief Runs the Group C sweep for one residual precision.
 *
 * A separate CSV file is produced for each residual precision. Every
 * requested condition number contributes exactly one row containing the
 * final normwise forward and backward errors whenever a valid approximate
 * solution is available. The common fields also record the termination
 * status, completed refinement updates, and final relative correction.
 *
 * @tparam T_residual Precision used to compute the residual.
 *
 * @param residual_precision_name Human-readable residual-precision name used
 *                                in the CSV metadata and filename.
 * @param kappas                  Common requested condition-number grid.
 */
template<class T_residual>
void run_residual_precision_sweep(
    std::string_view residual_precision_name,
    const std::vector<double>& kappas)
{
    using T_factor = hdnum::FP32;
    using T_work = hdnum::FP64;
    using T_measure = hdnum::FP256;

    mpir::TestProblemOptions problem_options;
    problem_options.rhs_mode =
        mpir::RightHandSideMode::random_normal_rhs;

    const mpir::ExperimentDescription experiment{
        mpir::ExperimentKind::residual_precision,
        mpir::MatrixFamily::random_spd,
        problem_dimension,
        {
            "fp32",
            "fp64",
            std::string(residual_precision_name),
            "fp256"
        }
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
    out << ",final_forward_error_inf,final_backward_error_inf\n";

    mpir::MixedIROptions<T_work> algorithm_options;
    algorithm_options.max_iterations = 20;
    algorithm_options.store_iterates = false;
    algorithm_options.detect_divergence = true;
    algorithm_options.scale_residual = true;
    algorithm_options.record_residual_diagnostics = false;

    constexpr std::string_view variant = "scaled";

    for (const double kappa : kappas) {
        // Reusing the fixed seeds and the same kappa grid for both runs makes
        // residual precision the only changed experimental variable.
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

        if (result.rel_corrections.size() != result.iterations) {
            throw std::logic_error(
                "Relative-correction count is inconsistent with completed updates"
            );
        }

        if (result.x.size() != 0
            && result.x.size() != problem_dimension) {
            throw std::logic_error(
                "Returned solution has an unexpected dimension"
            );
        }

        mpir::write_common_csv_fields(
            out,
            experiment,
            problem_options,
            algorithm_options,
            kappa,
            variant,
            result
        );

        out << ',';

        // An early failure can occur before x0 is constructed. Preserve its
        // status row while leaving both unavailable error fields empty.
        if (result.x.size() == problem_dimension) {
            const double final_forward_error =
                mpir::relative_forward_error_inf<T_measure>(
                    result.x,
                    problem.x_true
                );

            const double final_backward_error =
                mpir::normwise_backward_error_inf<T_measure>(
                    problem.A,
                    problem.b,
                    result.x
                );

            out << final_forward_error
                << ',' << final_backward_error;
        }
        else {
            out << ',';
        }

        out << '\n';

        std::cout
            << "fp32-fp64-" << residual_precision_name
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

    std::cout << "Writing results to: " << output_file << '\n';
}

} // namespace


int main()
{
    try {
        const auto kappas = group_c_kappas();

        // The two configurations requested by the supervisor. Factorization,
        // working, and measurement precisions remain fixed; only residual
        // precision changes.
        run_residual_precision_sweep<hdnum::FP64>(
            "fp64",
            kappas
        );

        run_residual_precision_sweep<hdnum::FP128>(
            "fp128",
            kappas
        );
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
