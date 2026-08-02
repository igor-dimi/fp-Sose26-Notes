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
constexpr double minimum_kappa = 1.0;
constexpr double max_boundary_multiple = 10.0;
constexpr std::size_t points_per_decade = 10;


/**
 * @brief Returns the standard dense Group B sweep for one factor precision.
 *
 * The grid begins at kappa = 1, uses ten logarithmic subdivisions per decade,
 * extends to 10/u_f, and contains the exact boundary 1/u_f.
 */
template<class T_factor>
[[nodiscard]] std::vector<double> group_b_kappas()
{
    return mpir::kappa_sweep<T_factor>(
        minimum_kappa,
        max_boundary_multiple,
        points_per_decade
    );
}


/**
 * @brief Runs the condition-number sweep for one precision configuration.
 *
 * A separate CSV file is produced for each configuration. Every requested
 * condition number contributes exactly one summary row containing the final
 * forward error when a valid approximate solution is available. The common
 * CSV fields already record the termination status, completed refinement
 * updates, and final relative correction.
 *
 * @tparam T_factor   Precision used for factorization and correction solves.
 * @tparam T_work     Precision used to store and update the iterate.
 * @tparam T_residual Precision used to compute the residual.
 * @tparam T_measure  Precision used to evaluate the reported error.
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
void run_condition_sweep(
    const mpir::PrecisionNames& precision_names,
    const std::vector<double>& kappas)
{
    mpir::TestProblemOptions problem_options;
    problem_options.rhs_mode =
        mpir::RightHandSideMode::random_normal_rhs;

    const mpir::ExperimentDescription experiment{
        mpir::ExperimentKind::condition_sweep,
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
    out << ",final_forward_error_inf\n";

    mpir::MixedIROptions<T_work> algorithm_options;
    algorithm_options.max_iterations = 20;
    algorithm_options.store_iterates = false;
    algorithm_options.detect_divergence = true;
    algorithm_options.scale_residual = true;
    algorithm_options.record_residual_diagnostics = false;

    constexpr std::string_view variant = "scaled";

    for (const double kappa : kappas) {
        // Reuse the same fixed seeds at every kappa and for every precision
        // configuration. Only the prescribed spectrum changes.
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

        // Early failures can occur before x0 is constructed. Keep their
        // status row, but leave the unavailable forward-error field empty.
        if (result.x.size() == problem_dimension) {
            const double final_forward_error =
                mpir::relative_forward_error_inf<T_measure>(
                    result.x,
                    problem.x_true
                );

            out << final_forward_error;
        }

        out << '\n';

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

        // FP64 is a same-precision control baseline. A boundary-scaled sweep
        // would approach the representational limit of FP64, so use the same
        // moderate grid as the Group A baseline.
        const std::vector<double> fp64_baseline_kappas{
            1.0,
            1.0e4,
            1.0e8,
            1.0e12
        };

        run_condition_sweep<
            hdnum::FP64,
            hdnum::FP64,
            hdnum::FP64,
            hdnum::FP256
        >(
            {"fp64", "fp64", "fp64", "fp256"},
            fp64_baseline_kappas
        );

        run_condition_sweep<
            hdnum::FP32,
            hdnum::FP64,
            hdnum::FP64,
            hdnum::FP256
        >(
            {"fp32", "fp64", "fp64", "fp256"},
            group_b_kappas<hdnum::FP32>()
        );

        run_condition_sweep<
            hdnum::FP32,
            hdnum::FP64,
            hdnum::FP128,
            hdnum::FP256
        >(
            {"fp32", "fp64", "fp128", "fp256"},
            group_b_kappas<hdnum::FP32>()
        );

        run_condition_sweep<
            hdnum::FP16,
            hdnum::FP64,
            hdnum::FP64,
            hdnum::FP256
        >(
            {"fp16", "fp64", "fp64", "fp256"},
            group_b_kappas<hdnum::FP16>()
        );

        run_condition_sweep<
            hdnum::FP16,
            hdnum::FP64,
            hdnum::FP128,
            hdnum::FP256
        >(
            {"fp16", "fp64", "fp128", "fp256"},
            group_b_kappas<hdnum::FP16>()
        );

        run_condition_sweep<
            hdnum::bfloat16,
            hdnum::FP64,
            hdnum::FP128,
            hdnum::FP256
        >(
            {"bfloat16", "fp64", "fp128", "fp256"},
            group_b_kappas<hdnum::bfloat16>()
        );

        // Optional stress-test configuration from the experiment plan.
        run_condition_sweep<
            FP8,
            hdnum::FP64,
            hdnum::FP128,
            hdnum::FP256
        >(
            {"fp8", "fp64", "fp128", "fp256"},
            group_b_kappas<FP8>()
        );
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
