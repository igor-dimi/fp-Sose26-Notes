#pragma once

/**
 * @file condition_grids.hpp
 * @brief Condition-number grids for iterative-refinement experiments.
 *
 * The utilities in this header generate condition numbers relative to
 * the approximate theoretical convergence boundary
 *
 *     kappa_* = 1 / u_f,
 *
 * where u_f is the unit roundoff of the factorization precision.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "hdnum_conversions.hpp"
#include "unit_roundoff.hpp"

namespace mpir {

namespace detail {

/**
 * @brief Computes the theoretical condition-number boundary for a
 *        factorization precision.
 *
 * @tparam T_factor Scalar type used for factorization.
 *
 * @return The boundary 1 / u_f represented as a double.
 *
 * @throws std::domain_error if the unit roundoff cannot be represented as
 *         a finite double in the interval (0, 1).
 * @throws std::overflow_error if 1 / u_f is not representable as a finite
 *         double.
 */
template<class T_factor>
[[nodiscard]] double factorization_kappa_boundary()
{
    const double unit_roundoff =
        scalar_cast<double>(
            default_unit_roundoff<T_factor>()
        );

    if (!std::isfinite(unit_roundoff)
        || unit_roundoff <= 0.0
        || unit_roundoff >= 1.0) {
        throw std::domain_error(
            "Factorization unit roundoff must be finite and lie in (0, 1)"
        );
    }

    const double boundary = 1.0 / unit_roundoff;

    if (!std::isfinite(boundary)) {
        throw std::overflow_error(
            "Factorization condition-number boundary exceeds double range"
        );
    }

    return boundary;
}


/**
 * @brief Sorts condition numbers and removes exact duplicates.
 */
inline void sort_and_remove_duplicates(
    std::vector<double>& kappas)
{
    std::sort(kappas.begin(), kappas.end());

    kappas.erase(
        std::unique(kappas.begin(), kappas.end()),
        kappas.end()
    );
}

} // namespace detail


/**
 * @brief Generates a small set of representative condition numbers.
 *
 * The returned grid contains a well-conditioned baseline and values below,
 * at, and above the approximate theoretical convergence boundary
 *
 *     kappa_* = 1 / u_f.
 *
 * More precisely, the function considers
 *
 *     1,
 *     0.01 kappa_*,
 *     0.1  kappa_*,
 *     0.5  kappa_*,
 *     kappa_*,
 *     2    kappa_*,
 *     10   kappa_*.
 *
 * Values below 1 are omitted because a matrix condition number cannot be
 * smaller than 1. The returned values are sorted and contain no duplicates.
 *
 * @tparam T_factor Scalar type used for factorization.
 *
 * @return Representative condition numbers in increasing order.
 *
 * @throws std::domain_error if the unit roundoff of T_factor is invalid.
 * @throws std::overflow_error if the boundary or one of its multiples
 *         exceeds the range of double.
 */
template<class T_factor>
[[nodiscard]] std::vector<double> representative_kappas()
{
    const double boundary =
        detail::factorization_kappa_boundary<T_factor>();

    constexpr double boundary_factors[] = {
        0.01,
        0.1,
        0.5,
        1.0,
        2.0,
        10.0,
        100.0
    };

    std::vector<double> kappas;
    kappas.reserve(7);

    // Well-conditioned baseline.
    kappas.push_back(1.0);

    for (const double factor : boundary_factors) {
        const double kappa = factor * boundary;

        if (!std::isfinite(kappa)) {
            throw std::overflow_error(
                "Representative condition number exceeds double range"
            );
        }

        if (kappa >= 1.0) {
            kappas.push_back(kappa);
        }
    }

    detail::sort_and_remove_duplicates(kappas);

    return kappas;
}


/**
 * @brief Generates a logarithmic condition-number sweep.
 *
 * The sweep starts at min_kappa and extends to
 *
 *     upper_boundary_factor / u_f,
 *
 * where u_f is the unit roundoff of the factorization precision.
 *
 * Consecutive regular grid points are separated by 1 / points_per_decade
 * decades. The upper endpoint is always included exactly. The theoretical
 * boundary 1 / u_f is also inserted exactly whenever it lies within the
 * requested interval.
 *
 * @tparam T_factor Scalar type used for factorization.
 *
 * @param min_kappa            Smallest condition number in the sweep.
 *                             Must be finite and at least 1.
 * @param upper_boundary_factor
 *                             Upper endpoint as a multiple of 1 / u_f.
 *                             Must be finite and positive.
 * @param points_per_decade    Number of logarithmic subdivisions per
 *                             decade. Must be positive.
 *
 * @return Sorted condition numbers without exact duplicates.
 *
 * @throws std::invalid_argument if an argument is invalid or the computed
 *         upper endpoint is smaller than min_kappa.
 * @throws std::domain_error if the unit roundoff of T_factor is invalid.
 * @throws std::overflow_error if the requested range exceeds double range.
 * @throws std::length_error if the requested grid is too large.
 */
template<class T_factor>
[[nodiscard]] std::vector<double> kappa_sweep(
    double min_kappa = 1.0,
    double upper_boundary_factor = 10.0,
    std::size_t points_per_decade = 10)
{
    if (!std::isfinite(min_kappa) || min_kappa < 1.0) {
        throw std::invalid_argument(
            "min_kappa must be finite and at least 1"
        );
    }

    if (!std::isfinite(upper_boundary_factor)
        || upper_boundary_factor <= 0.0) {
        throw std::invalid_argument(
            "upper_boundary_factor must be finite and positive"
        );
    }

    if (points_per_decade == 0) {
        throw std::invalid_argument(
            "points_per_decade must be positive"
        );
    }

    const double boundary =
        detail::factorization_kappa_boundary<T_factor>();

    const double max_kappa =
        upper_boundary_factor * boundary;

    if (!std::isfinite(max_kappa)) {
        throw std::overflow_error(
            "Upper condition-number boundary exceeds double range"
        );
    }

    if (max_kappa < min_kappa) {
        throw std::invalid_argument(
            "Computed upper boundary is smaller than min_kappa"
        );
    }

    if (max_kappa == min_kappa) {
        return {min_kappa};
    }

    const double number_of_decades =
        std::log10(max_kappa / min_kappa);

    const double required_steps = std::ceil(
        number_of_decades
        * static_cast<double>(points_per_decade)
    );

    const double maximum_steps =
        static_cast<double>(
            std::numeric_limits<std::size_t>::max() - 2
        );

    if (!std::isfinite(required_steps)
        || required_steps > maximum_steps) {
        throw std::length_error(
            "Requested condition-number sweep is too large"
        );
    }

    const std::size_t regular_steps =
        static_cast<std::size_t>(required_steps);

    std::vector<double> kappas;
    kappas.reserve(regular_steps + 2);

    for (std::size_t i = 0; i < regular_steps; ++i) {
        const double exponent =
            static_cast<double>(i)
            / static_cast<double>(points_per_decade);

        const double kappa =
            min_kappa * std::pow(10.0, exponent);

        if (!std::isfinite(kappa)) {
            throw std::overflow_error(
                "Generated condition number exceeds double range"
            );
        }

        // Rounding near the endpoint can occasionally produce max_kappa
        // one step early. The exact endpoint is appended below.
        if (kappa < max_kappa) {
            kappas.push_back(kappa);
        }
    }

    // Include both the requested endpoint and the theoretical boundary
    // exactly rather than relying on logarithmic stepping to reach them.
    kappas.push_back(max_kappa);

    if (boundary >= min_kappa && boundary <= max_kappa) {
        kappas.push_back(boundary);
    }

    detail::sort_and_remove_duplicates(kappas);

    return kappas;
}

} // namespace mpir