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
 * @brief Result of one working-precision direct LU solve.
 */
template<class T>
struct DirectSolveResult {
    hdnum::Vector<T> x;
    std::string_view status = "failure";
};


/**
 * @brief Returns the condition-number grid used by Group D.
 *
 * The mixed method factors in FP32. The grid therefore begins at kappa = 1,
 * uses ten logarithmic subdivisions per decade, extends to 10/u_f, and
 * contains the exact theoretical boundary 1/u_f.
 */
[[nodiscard]] std::vector<double> group_d_kappas()
{
    return mpir::kappa_sweep<hdnum::FP32>(
        minimum_kappa,
        max_boundary_multiple,
        points_per_decade
    );
}


/**
 * @brief Solves A*x = b using full-pivoting LU entirely in T.
 *
 * The matrix, factorization, triangular solves, and returned solution all use
 * the same precision. This is the direct working-precision baseline against
 * which mixed-precision iterative refinement is compared.
 */
template<class T>
[[nodiscard]] DirectSolveResult<T> direct_lu_solve(
    const hdnum::DenseMatrix<T>& A,
    const hdnum::Vector<T>& b)
{
    if (A.rowsize() == 0 || A.rowsize() != A.colsize()) {
        throw std::invalid_argument(
            "direct_lu_solve: A must be square and nonempty"
        );
    }

    if (A.rowsize() != b.size()) {
        throw std::invalid_argument(
            "direct_lu_solve: incompatible matrix and right-hand-side sizes"
        );
    }

    DirectSolveResult<T> result;

    if (!mpir::all_finite(A) || !mpir::all_finite(b)) {
        result.status = "input-non-finite";
        return result;
    }

    const std::size_t n = b.size();
    hdnum::DenseMatrix<T> LU(A);
    hdnum::Vector<std::size_t> p(n);
    hdnum::Vector<std::size_t> q(n);

    hdnum::lr_fullpivot(LU, p, q);

    if (!mpir::all_finite(LU)) {
        result.status = "factorization-non-finite";
        return result;
    }

    result.x = mpir::solve_with_lu_fullpivot(LU, p, q, b);

    if (!mpir::all_finite(result.x)) {
        result.x = hdnum::Vector<T>();
        result.status = "solution-non-finite";
        return result;
    }

    result.status = "success";
    return result;
}


/**
 * @brief Writes a direct-solve row using the common experiment CSV schema.
 *
 * Iterative-refinement-only fields are deliberately left empty. In
 * particular, a direct solve has no stopping tolerance, convergence flag,
 * refinement iteration count, or relative correction.
 */
void write_direct_common_fields(
    std::ostream& out,
    const mpir::ExperimentDescription& experiment,
    const mpir::TestProblemOptions& problem_options,
    double requested_kappa,
    std::string_view status)
{
    mpir::write_csv_text(out, mpir::to_string(experiment.kind));
    out << ',';

    mpir::write_csv_text(out, mpir::to_string(experiment.matrix_family));
    out << ',' << experiment.dimension << ',';

    mpir::write_csv_text(out, "fp64");
    out << ',';
    mpir::write_csv_text(out, "fp64");
    out << ',';
    mpir::write_csv_text(out, "none");
    out << ',';
    mpir::write_csv_text(out, experiment.precisions.measure);
    out << ',';
    mpir::write_csv_text(out, "direct-lu");
    out << ',';

    mpir::write_csv_text(out, mpir::to_string(problem_options.rhs_mode));
    out
        << ',' << problem_options.matrix_seed_u
        << ',' << problem_options.matrix_seed_v
        << ',' << problem_options.vector_seed
        << ',' << problem_options.rotation_theta
        << ',' << requested_kappa;

    // Empty common fields:
    // max_iterations, effective_rel_correction_tol, detect_divergence,
    // divergence_growth_factor, divergence_growth_steps, store_iterates,
    // scale_residual, record_residual_diagnostics.
    out << ",,,,,,,,,";

    mpir::write_csv_text(out, status);

    // converged and final_rel_correction are not applicable. A direct solve
    // performs zero refinement updates.
    out << ",,0,";
}


/**
 * @brief Runs Group D: FP32-FP64-FP128 IR versus direct FP64 LU.
 */
void run_direct_solve_comparison()
{
    using T_factor = hdnum::FP32;
    using T_work = hdnum::FP64;
    using T_residual = hdnum::FP128;
    using T_measure = hdnum::FP256;

    mpir::TestProblemOptions problem_options;
    problem_options.rhs_mode =
        mpir::RightHandSideMode::random_normal_rhs;

    // The precision names identify the mixed method in the filename. Rows for
    // the direct baseline explicitly record FP64 factor and work precision.
    const mpir::ExperimentDescription experiment{
        mpir::ExperimentKind::direct_solve_comparison,
        mpir::MatrixFamily::random_spd,
        problem_dimension,
        {"fp32", "fp64", "fp128", "fp256"}
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
            problem_options,
            "vs-direct-fp64"
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

    constexpr std::string_view mixed_variant = "mixed-ir";

    for (const double kappa : group_d_kappas()) {
        // Construct the problem once. Both methods therefore receive exactly
        // the same working-precision A and b and share the same FP256
        // reference solution.
        const auto problem =
            mpir::make_random_spd_problem<T_work, T_measure>(
                problem_dimension,
                kappa,
                problem_options
            );

        const auto mixed_result =
            mpir::mixed_ir<T_factor, T_work, T_residual>(
                problem.A,
                problem.b,
                algorithm_options
            );

        if (mixed_result.rel_corrections.size()
            != mixed_result.iterations) {
            throw std::logic_error(
                "Relative-correction count is inconsistent with completed updates"
            );
        }

        if (mixed_result.x.size() != 0
            && mixed_result.x.size() != problem_dimension) {
            throw std::logic_error(
                "Mixed-IR solution has an unexpected dimension"
            );
        }

        const auto direct_result =
            direct_lu_solve(problem.A, problem.b);

        // Mixed-IR row.
        mpir::write_common_csv_fields(
            out,
            experiment,
            problem_options,
            algorithm_options,
            kappa,
            mixed_variant,
            mixed_result
        );

        out << ',';

        if (mixed_result.x.size() == problem_dimension) {
            const double forward_error =
                mpir::relative_forward_error_inf<T_measure>(
                    mixed_result.x,
                    problem.x_true
                );

            const double backward_error =
                mpir::normwise_backward_error_inf<T_measure>(
                    problem.A,
                    problem.b,
                    mixed_result.x
                );

            out << forward_error << ',' << backward_error;
        }
        else {
            out << ',';
        }

        out << '\n';

        // Direct-FP64 row.
        write_direct_common_fields(
            out,
            experiment,
            problem_options,
            kappa,
            direct_result.status
        );

        // Terminate the empty direct-solve final_rel_correction field before
        // appending the two experiment-specific error columns.
        out << ',';

        if (direct_result.x.size() == problem_dimension) {
            const double forward_error =
                mpir::relative_forward_error_inf<T_measure>(
                    direct_result.x,
                    problem.x_true
                );

            const double backward_error =
                mpir::normwise_backward_error_inf<T_measure>(
                    problem.A,
                    problem.b,
                    direct_result.x
                );

            out << forward_error << ',' << backward_error;
        }
        else {
            out << ',';
        }

        out << '\n';

        std::cout
            << "kappa = " << kappa
            << ", mixed status = "
            << mpir::to_string(mixed_result.status)
            << ", mixed iterations = " << mixed_result.iterations
            << ", direct status = " << direct_result.status
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
        run_direct_solve_comparison();
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
