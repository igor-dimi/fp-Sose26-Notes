#pragma once

/**
 * @file experiment_io.hpp
 * @brief Shared descriptions and I/O utilities for numerical experiments.
 *
 * This header defines the common metadata used to identify experiments,
 * matrix families, problem dimensions, and precision configurations. It also
 * provides stable textual identifiers, filename generation, output-directory
 * construction, and common CSV-writing utilities.
 */

#include <cstddef>
#include <string>
#include <filesystem>
#include <ostream>
#include <string_view>
#include <stdexcept>

#include "mixed_ir.hpp"
#include "test_matrices.hpp"

namespace mpir {

/**
 * @brief Identifies the purpose and output structure of an experiment.
 */
enum class ExperimentKind {
    /// Records errors and corrections at every refinement iteration.
    convergence_history,

    /// Studies algorithm behavior over a range of condition numbers.
    condition_sweep,

    /// Compares iterative refinement with different residual precisions.
    residual_precision,

    /// Compares mixed-precision refinement with a direct working-precision solve.
    direct_solve_comparison,

    /// Compares iterative refinement with and without residual scaling.
    residual_scaling,

    /// Studies failures caused by the limited numerical range of a precision.
    range_failure
};

/**
 * @brief Identifies the family used to construct the test matrix.
 */
enum class MatrixFamily {
    /// Random symmetric positive-definite matrix with a prescribed spectrum.
    random_spd,

    /// Symmetric positive-definite matrix constructed using controlled rotations.
    rotated_spd,

    /// General matrix constructed from a random singular value decomposition.
    random_svd
};

/**
 * @brief Human-readable names of the precisions used by an experiment.
 *
 * These names are used in CSV metadata and automatically generated filenames.
 * Suitable values include `"fp16"`, `"fp32"`, `"fp64"`, and `"fp128"`.
 */
struct PrecisionNames {
    /// Precision used for matrix factorization and triangular solves.
    std::string factor;

    /// Precision used to store and update the approximate solution.
    std::string work;

    /// Precision used to compute the residual.
    std::string residual;

    /// Precision used to evaluate errors and other reported quantities.
    std::string measure;
};

/**
 * @brief Metadata that identifies an experiment independently of a single run.
 *
 * Run-specific information, such as the prescribed condition number,
 * random seeds, algorithm options, and termination result, is supplied
 * separately when writing CSV output.
 */
struct ExperimentDescription {
    /// Purpose and output category of the experiment.
    ExperimentKind kind;

    /// Family used to generate the test matrix.
    MatrixFamily matrix_family;

    /// Dimension of the square test matrix.
    std::size_t dimension;

    /// Names of the precisions used in the experiment.
    PrecisionNames precisions;
};


/**
 * @brief Returns the stable textual identifier of an experiment kind.
 *
 * The returned identifiers are suitable for CSV metadata, directory names,
 * and automatically generated filenames.
 *
 * @param kind Experiment kind to convert.
 * @return Stable textual identifier.
 *
 * @throws std::invalid_argument if kind does not contain a valid enumerator.
 */
[[nodiscard]] inline std::string_view
to_string(ExperimentKind kind)
{
    switch (kind) {
        case ExperimentKind::convergence_history:
            return "convergence-history";

        case ExperimentKind::condition_sweep:
            return "condition-sweep";

        case ExperimentKind::residual_precision:
            return "residual-precision";

        case ExperimentKind::direct_solve_comparison:
            return "direct-solve-comparison";

        case ExperimentKind::residual_scaling:
            return "residual-scaling";

        case ExperimentKind::range_failure:
            return "range-failure";
    }

    throw std::invalid_argument("Invalid ExperimentKind");
}


/**
 * @brief Returns the stable textual identifier of a matrix family.
 *
 * @param family Matrix family to convert.
 * @return Stable textual identifier.
 *
 * @throws std::invalid_argument if family does not contain a valid enumerator.
 */
[[nodiscard]] inline std::string_view
to_string(MatrixFamily family)
{
    switch (family) {
        case MatrixFamily::random_spd:
            return "random-spd";

        case MatrixFamily::rotated_spd:
            return "rotated-spd";

        case MatrixFamily::random_svd:
            return "random-svd";
    }

    throw std::invalid_argument("Invalid MatrixFamily");
}


/**
 * @brief Returns the stable textual identifier of a right-hand-side mode.
 *
 * The identifiers distinguish modes that construct a known solution from
 * the mode that generates the right-hand side directly.
 *
 * @param mode Right-hand-side mode to convert.
 * @return Stable textual identifier.
 *
 * @throws std::invalid_argument if mode does not contain a valid enumerator.
 */
[[nodiscard]] inline std::string_view
to_string(RightHandSideMode mode)
{
    switch (mode) {
        case RightHandSideMode::ones_solution:
            return "ones-solution";

        case RightHandSideMode::random_sign_solution:
            return "random-sign-solution";

        case RightHandSideMode::random_normal_rhs:
            return "random-normal";
    }

    throw std::invalid_argument("Invalid RightHandSideMode");
}


/**
 * @brief Returns the stable textual identifier of an iterative-refinement
 *        termination status.
 *
 * @param status Termination status to convert.
 * @return Stable textual identifier.
 *
 * @throws std::invalid_argument if status does not contain a valid enumerator.
 */
[[nodiscard]] inline std::string_view
to_string(MixedIRStatus status)
{
    switch (status) {
        case MixedIRStatus::converged:
            return "converged";

        case MixedIRStatus::max_iterations:
            return "max-iterations";

        case MixedIRStatus::factorization_input_non_finite:
            return "factorization-input-non-finite";

        case MixedIRStatus::non_finite:
            return "non-finite";

        case MixedIRStatus::diverged:
            return "diverged";
    }

    throw std::invalid_argument("Invalid MixedIRStatus");
}

/**
 * @brief Applies the filename-component validation used by this header.
 *
 * Empty components and path separators are rejected. The function does not
 * modify or normalize the supplied component.
 *
 * @param component Component to validate.
 * @param name      Name used to identify the component in error messages.
 *
 * @throws std::invalid_argument if the component is empty or contains a path
 *         separator.
 */
inline void validate_filename_component(
    std::string_view component,
    std::string_view name)
{
    if (component.empty()) {
        throw std::invalid_argument(
            std::string(name) + " must not be empty"
        );
    }

    if (component.find_first_of("/\\") != std::string_view::npos) {
        throw std::invalid_argument(
            std::string(name) + " must not contain path separators"
        );
    }
}


/**
 * @brief Generates the CSV filename for an experiment.
 *
 * The filename encodes the experiment kind, matrix family, dimension,
 * right-hand-side mode, and precision configuration. An optional tag can
 * distinguish variants or comparisons belonging to the same experiment.
 *
 * The generated filename has the form shown below. It is split across lines
 * only for readability.
 *
 * @code
 * <experiment>__<matrix>__n-<dimension>__<rhs>
 * __uf-<factor>__u-<work>__ur-<residual>__um-<measure>
 * __<tag>.csv
 * @endcode
 *
 * The tag component is omitted when tag is empty.
 *
 * @param experiment      Common experiment description.
 * @param problem_options Test-problem configuration.
 * @param tag             Optional filename tag, such as `"comparison"`.
 *
 * @return Generated filename, including the `.csv` extension.
 *
 * @throws std::invalid_argument if an enum value is invalid, or if a precision
 *         name or nonempty tag is not a valid filename component.
 */
[[nodiscard]] inline std::string make_experiment_filename(
    const ExperimentDescription& experiment,
    const TestProblemOptions& problem_options,
    std::string_view tag = {})
{
    validate_filename_component(
        experiment.precisions.factor,
        "Factorization precision name"
    );
    validate_filename_component(
        experiment.precisions.work,
        "Working precision name"
    );
    validate_filename_component(
        experiment.precisions.residual,
        "Residual precision name"
    );
    validate_filename_component(
        experiment.precisions.measure,
        "Measurement precision name"
    );

    if (!tag.empty()) {
        validate_filename_component(tag, "Filename tag");
    }

    std::string filename;

    filename += to_string(experiment.kind);
    filename += "__";
    filename += to_string(experiment.matrix_family);
    filename += "__n-";
    filename += std::to_string(experiment.dimension);
    filename += "__";
    filename += to_string(problem_options.rhs_mode);
    filename += "__uf-";
    filename += experiment.precisions.factor;
    filename += "__u-";
    filename += experiment.precisions.work;
    filename += "__ur-";
    filename += experiment.precisions.residual;
    filename += "__um-";
    filename += experiment.precisions.measure;

    if (!tag.empty()) {
        filename += "__";
        filename += tag;
    }

    filename += ".csv";

    return filename;
}

/**
 * @brief Constructs and creates the raw-data directory for an experiment.
 *
 * The supplied root is expected to denote the raw-results directory, for
 * example `results/raw`. The experiment kind determines the corresponding
 * subdirectory.
 *
 * Existing directories are left unchanged. Missing parent directories are
 * created recursively.
 *
 * @param raw_results_root Root directory for raw experimental results.
 * @param kind             Kind of experiment whose directory is required.
 *
 * @return Path to the experiment-specific output directory.
 *
 * @throws std::invalid_argument if kind does not contain a valid enumerator.
 * @throws std::filesystem::filesystem_error if directory creation fails.
 */
[[nodiscard]] inline std::filesystem::path make_output_directory(
    const std::filesystem::path& raw_results_root,
    ExperimentKind kind
)
{
    std::filesystem::path output_directory;

    switch (kind) {
        case ExperimentKind::convergence_history:
            output_directory =
                raw_results_root / "convergence";
            break;

        case ExperimentKind::condition_sweep:
            output_directory =
                raw_results_root / "condition_sweeps";
            break;

        case ExperimentKind::residual_precision:
            output_directory =
                raw_results_root / "residual_precision";
            break;

        case ExperimentKind::direct_solve_comparison:
            output_directory =
                raw_results_root / "direct_comparison";
            break;

        case ExperimentKind::residual_scaling:
            output_directory =
                raw_results_root / "robustness" / "residual_scaling";
            break;

        case ExperimentKind::range_failure:
            output_directory =
                raw_results_root / "robustness" / "range_failures";
            break;

        default:
            throw std::invalid_argument("Invalid ExperimentKind");
    }

    std::filesystem::create_directories(output_directory);
    return output_directory;
}

/**
 * @brief Writes the common CSV column names shared by all experiments.
 *
 * The function writes neither a trailing comma nor a newline. The caller can
 * therefore append experiment-specific columns before terminating the header.
 *
 * The order of these columns must exactly match the order used by
 * write_common_csv_fields().
 *
 * @param out Output stream receiving the CSV header.
 */
inline void write_common_csv_header(std::ostream& out)
{
    out
        << "experiment,"
        << "matrix_family,"
        << "dimension,"
        << "factor_precision,"
        << "work_precision,"
        << "residual_precision,"
        << "measure_precision,"
        << "variant,"
        << "rhs_mode,"
        << "matrix_seed_u,"
        << "matrix_seed_v,"
        << "vector_seed,"
        << "rotation_theta,"
        << "requested_kappa,"
        << "max_iterations,"
        << "effective_rel_correction_tol,"
        << "detect_divergence,"
        << "divergence_growth_factor,"
        << "divergence_growth_steps,"
        << "store_iterates,"
        << "scale_residual,"
        << "record_residual_diagnostics,"
        << "status,"
        << "converged,"
        << "total_iterations,"
        << "final_rel_correction";
}

/**
 * @brief Writes one escaped CSV text field.
 *
 * The field is enclosed in double quotes. Embedded double quotes are
 * represented by two consecutive double quotes.
 *
 * @param out Output stream receiving the field.
 * @param value Text to quote and escape.
 */
inline void write_csv_text(
    std::ostream& out,
    std::string_view value)
{
    out.put('"');

    for (const char character : value) {
        if (character == '"') {
            out << "\"\"";
        }
        else {
            out.put(character);
        }
    }

    out.put('"');
}


/**
 * @brief Writes the common metadata and result fields for one CSV row.
 *
 * The values are written in exactly the same order as the column names
 * produced by write_common_csv_header().
 *
 * The function writes neither a leading comma, a trailing comma, nor a
 * newline. The caller can therefore append experiment-specific values.
 *
 * Run-level values, such as the termination status and final relative
 * correction, are repeated in every row of a convergence-history file.
 * The effective correction tolerance is the configured positive tolerance,
 * or the working-precision unit roundoff otherwise. Boolean values are written
 * as 0 or 1.
 *
 * @tparam T_work Working-precision scalar type.
 *
 * @param out               Output stream receiving the CSV fields.
 * @param experiment        Common experiment description.
 * @param problem_options   Test-problem configuration.
 * @param algorithm_options Iterative-refinement configuration.
 * @param requested_kappa   Condition number requested from the generator.
 * @param variant           Algorithm variant, such as `"scaled"` or
 *                          `"unscaled"`.
 * @param result            Result of the iterative-refinement run.
 *
 * @throws std::invalid_argument if an enum value is invalid.
 * @note Numeric formatting follows the current state of out.
 */
template<class T_work>
void write_common_csv_fields(
    std::ostream& out,
    const ExperimentDescription& experiment,
    const TestProblemOptions& problem_options,
    const MixedIROptions<T_work>& algorithm_options,
    double requested_kappa,
    std::string_view variant,
    const MixedIRResult<T_work>& result)
{
    const T_work effective_rel_correction_tol =
        algorithm_options.rel_correction_tol > T_work(0)
            ? algorithm_options.rel_correction_tol
            : default_unit_roundoff<T_work>();

    write_csv_text(out, to_string(experiment.kind));
    out << ',';

    write_csv_text(out, to_string(experiment.matrix_family));
    out << ','
        << experiment.dimension
        << ',';

    write_csv_text(out, experiment.precisions.factor);
    out << ',';

    write_csv_text(out, experiment.precisions.work);
    out << ',';

    write_csv_text(out, experiment.precisions.residual);
    out << ',';

    write_csv_text(out, experiment.precisions.measure);
    out << ',';

    write_csv_text(out, variant);
    out << ',';

    write_csv_text(out, to_string(problem_options.rhs_mode));
    out
        << ',' << problem_options.matrix_seed_u
        << ',' << problem_options.matrix_seed_v
        << ',' << problem_options.vector_seed
        << ',' << problem_options.rotation_theta
        << ',' << requested_kappa
        << ',' << algorithm_options.max_iterations
        << ',' << effective_rel_correction_tol
        << ',' << (algorithm_options.detect_divergence ? 1 : 0)
        << ',' << algorithm_options.divergence_growth_factor
        << ',' << algorithm_options.divergence_growth_steps
        << ',' << (algorithm_options.store_iterates ? 1 : 0)
        << ',' << (algorithm_options.scale_residual ? 1 : 0)
        << ',' << (
            algorithm_options.record_residual_diagnostics ? 1 : 0
        )
        << ',';

    write_csv_text(out, to_string(result.status));
    out
        << ',' << (result.converged() ? 1 : 0)
        << ',' << result.iterations
        << ',' << result.final_rel_correction;
}



} // namespace mpir
