#!/usr/bin/env python3
"""Plot Group A mixed-precision iterative-refinement histories.

By default, the script reads every convergence-history CSV from
``results/raw/convergence`` and writes one two-panel figure per precision
configuration to ``results/plots/convergence``. Pass
``--include-relative-correction`` to add the relative-correction history as a
third panel.

The intended location of this script is ``code/scripts``. Input files or
directories may also be supplied explicitly on the command line.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mpir_plotting.csv_validation import (
    coerce_numeric_columns,
    invariant_value,
    read_csv_checked,
    require_invariant_columns,
)
from mpir_plotting.paths import (
    discover_csv_files,
    mirrored_plot_path,
    resolve_results_roots,
)
from mpir_plotting.precisions import (
    factorization_boundary,
    precision_label,
)
from mpir_plotting.styles import (
    MIXED_IR_STATUS_NAMES,
    status_code,
    status_key,
)


CSV_PATTERN = "convergence-history__*.csv"
DEFAULT_RAW_SUBDIRECTORY = Path("convergence")

REQUIRED_COLUMNS = {
    "requested_kappa",
    "iteration",
    "forward_error_inf",
    "backward_error_inf",
    "rel_correction",
    "factor_precision",
    "work_precision",
    "residual_precision",
    "measure_precision",
    "matrix_family",
    "dimension",
    "variant",
    "status",
}

NUMERIC_COLUMNS = (
    "requested_kappa",
    "iteration",
    "forward_error_inf",
    "backward_error_inf",
    "rel_correction",
)

INVARIANT_METADATA_COLUMNS = (
    "factor_precision",
    "work_precision",
    "residual_precision",
    "measure_precision",
    "matrix_family",
    "dimension",
    "variant",
)

DEFAULT_METRICS = (
    (
        "forward_error_inf",
        "(a) Relative forward error",
        "Relative forward error, infinity norm",
    ),
    (
        "backward_error_inf",
        "(b) Normwise backward error",
        "Normwise backward error, infinity norm",
    ),
)

RELATIVE_CORRECTION_METRIC = (
    "rel_correction",
    "(c) Relative correction",
    "Relative correction norm",
)

ALL_METRICS = DEFAULT_METRICS + (RELATIVE_CORRECTION_METRIC,)

REPRESENTATIVE_BOUNDARY_FACTORS = (
    0.01,
    0.1,
    0.5,
    1.0,
    2.0,
    10.0,
)

# Full decimal notation is easier to read for the FP8, bfloat16, FP16,
# and FP32 representative grids. Scientific notation avoids excessively
# wide labels for genuinely large values such as the FP64 baseline.
DECIMAL_NOTATION_LIMIT = 1.0e9


def parse_arguments() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description=(
            "Plot all Group A convergence-history CSV files as two-panel "
            "figures, optionally including relative-correction histories."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Optional CSV files or directories. Directories are searched "
            f"recursively for {CSV_PATTERN}. If omitted, all matching files "
            "under results/raw/convergence are plotted."
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help=(
            "Root of the raw-results tree. Defaults to "
            "<repository>/results/raw."
        ),
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=None,
        help=(
            "Root of the plot-results tree. Defaults to "
            "<repository>/results/plots."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Output format (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Raster resolution for PNG output (default: 200).",
    )
    parser.add_argument(
        "--include-relative-correction",
        action="store_true",
        help=(
            "Add relative correction as a third panel. By default, only "
            "forward and backward errors are plotted."
        ),
    )

    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def read_history(csv_path: Path) -> pd.DataFrame:
    """Read and validate one convergence-history CSV file."""
    dataframe = read_csv_checked(csv_path, REQUIRED_COLUMNS)
    coerce_numeric_columns(dataframe, NUMERIC_COLUMNS, csv_path)

    if dataframe["requested_kappa"].isna().any():
        raise ValueError(
            f"{csv_path} contains a missing or invalid requested_kappa."
        )
    if (
        ~np.isfinite(dataframe["requested_kappa"])
        | (dataframe["requested_kappa"] <= 0.0)
    ).any():
        raise ValueError(
            f"{csv_path} contains a non-positive or non-finite "
            "requested_kappa."
        )

    if dataframe["status"].isna().any():
        raise ValueError(f"{csv_path} contains a missing status.")

    dataframe["status"] = (
        dataframe["status"].astype(str).str.strip().str.lower()
    )
    unknown_statuses = sorted(
        set(dataframe["status"]) - MIXED_IR_STATUS_NAMES
    )
    if unknown_statuses:
        unknown = ", ".join(repr(status) for status in unknown_statuses)
        expected = ", ".join(sorted(MIXED_IR_STATUS_NAMES))
        raise ValueError(
            f"{csv_path} contains unknown status value(s): {unknown}. "
            f"Expected one of: {expected}."
        )

    status_counts = dataframe.groupby("requested_kappa")["status"].nunique()
    inconsistent_kappas = status_counts[status_counts != 1].index.tolist()
    if inconsistent_kappas:
        formatted_kappas = ", ".join(
            f"{kappa:g}" for kappa in inconsistent_kappas
        )
        raise ValueError(
            f"{csv_path} contains more than one termination status for "
            f"kappa = {formatted_kappas}."
        )

    require_invariant_columns(
        dataframe,
        INVARIANT_METADATA_COLUMNS,
        csv_path,
    )

    history = dataframe.dropna(subset=["iteration"]).copy()
    status_only_kappas = sorted(
        dataframe.loc[
            dataframe["iteration"].isna(),
            "requested_kappa",
        ].unique()
    )
    if status_only_kappas:
        formatted_kappas = ", ".join(
            f"{kappa:g}" for kappa in status_only_kappas
        )
        warnings.warn(
            f"{csv_path.name}: no iterate is available for kappa = "
            f"{formatted_kappas}; these status-only runs cannot be plotted.",
            stacklevel=1,
        )

    if history.empty:
        raise ValueError(f"{csv_path} contains no available iterates to plot.")

    nonintegral_iterations = (
        ~np.isfinite(history["iteration"])
        | (history["iteration"] < 0.0)
        | (
            (history["iteration"] % 1.0).abs()
            > np.finfo(float).eps
        )
    )
    if nonintegral_iterations.any():
        raise ValueError(
            f"{csv_path} contains negative, non-finite, or nonintegral "
            "iteration values."
        )

    history["iteration"] = history["iteration"].astype(int)
    duplicates = history.duplicated(
        subset=["requested_kappa", "iteration"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError(
            f"{csv_path} contains duplicate rows for a condition number "
            "and iteration."
        )

    missing_metrics = history[
        [column for column, _, _ in ALL_METRICS]
    ].isna()
    if missing_metrics[
        ["forward_error_inf", "backward_error_inf"]
    ].any().any():
        raise ValueError(
            f"{csv_path} contains missing forward or backward errors for "
            "an available iterate."
        )

    if history.loc[history["iteration"] > 0, "rel_correction"].isna().any():
        raise ValueError(
            f"{csv_path} contains a missing relative correction after "
            "iteration 0."
        )

    return history.sort_values(
        ["requested_kappa", "iteration"]
    ).reset_index(drop=True)


def decimal_math_text(value: float) -> str:
    """Format a moderate finite value as readable decimal math text."""
    text = np.format_float_positional(
        value,
        precision=12,
        unique=False,
        fractional=False,
        trim="-",
    )
    integer_part, separator, fractional_part = text.partition(".")
    grouped_integer = f"{int(integer_part):,}".replace(",", "{,}")

    if not separator:
        return grouped_integer

    return f"{grouped_integer}.{fractional_part}"


def scientific_math_text(value: float) -> str:
    """Format a large finite value compactly as mathematical notation."""
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0**exponent

    if np.isclose(coefficient, 1.0, rtol=1.0e-12, atol=0.0):
        return f"10^{{{exponent}}}"

    coefficient_text = np.format_float_positional(
        coefficient,
        precision=6,
        unique=False,
        fractional=False,
        trim="-",
    )
    return fr"{coefficient_text}\times10^{{{exponent}}}"


def number_math_text(value: float) -> str:
    """Choose readable decimal or compact scientific notation."""
    if not np.isfinite(value):
        return f"{value:g}"

    if abs(value) < DECIMAL_NOTATION_LIMIT:
        return decimal_math_text(value)

    return scientific_math_text(value)


def representative_boundary_factor(
    kappa: float,
    boundary: float,
) -> float | None:
    """Recognize a representative-grid point as a multiple of kappa_* ."""
    if np.isclose(kappa, 1.0, rtol=1.0e-12, atol=1.0e-15):
        return None

    ratio = kappa / boundary
    for factor in REPRESENTATIVE_BOUNDARY_FACTORS:
        if np.isclose(ratio, factor, rtol=1.0e-10, atol=1.0e-14):
            return factor

    return None


def boundary_factor_math_text(factor: float) -> str:
    """Format a recognized multiple of the factorization boundary."""
    if np.isclose(factor, 1.0):
        return r"\kappa_*"

    factor_text = decimal_math_text(factor)
    return fr"{factor_text}\kappa_*"


def kappa_label(kappa: float, boundary: float) -> str:
    """Format one condition number and expose its boundary-grid role."""
    value_text = number_math_text(kappa)
    factor = representative_boundary_factor(kappa, boundary)

    if factor is not None:
        factor_text = boundary_factor_math_text(factor)
        return fr"${factor_text}={value_text}$"

    if np.isclose(kappa, 1.0, rtol=1.0e-12, atol=1.0e-15):
        return fr"$\kappa={value_text}$ (baseline)"

    return fr"$\kappa={value_text}$"


def run_status(group: pd.DataFrame) -> str:
    """Return the single validated termination status for one run."""
    statuses = group["status"].unique()
    if len(statuses) != 1:
        raise ValueError(
            "A condition-number history must have exactly one status."
        )
    return str(statuses[0])


def legend_title(boundary: float, statuses: Iterable[str]) -> str:
    """Construct the boundary definition and status key for the legend."""
    boundary_text = number_math_text(boundary)
    return "\n".join(
        (
            fr"$\kappa_*=1/u_f={boundary_text}$",
            status_key(statuses),
        )
    )


def figure_title(dataframe: pd.DataFrame, csv_path: Path) -> str:
    """Construct a title from the invariant CSV metadata."""
    factor = str(invariant_value(dataframe, "factor_precision", csv_path))
    work = str(invariant_value(dataframe, "work_precision", csv_path))
    residual = str(
        invariant_value(dataframe, "residual_precision", csv_path)
    )
    dimension = int(invariant_value(dataframe, "dimension", csv_path))
    variant = str(invariant_value(dataframe, "variant", csv_path))

    precisions = (
        f"{precision_label(factor)}–{precision_label(work)}–"
        f"{precision_label(residual)}"
    )
    return (
        f"Convergence histories: {precisions} "
        f"(n = {dimension}, {variant})"
    )


def positive_log_values(values: pd.Series) -> pd.Series:
    """Replace nonpositive values by NaN for a logarithmic axis."""
    return values.where(values > 0.0)


def configure_axis(axis: Axes, title: str, ylabel: str) -> None:
    """Apply common formatting to one history panel."""
    axis.set_yscale("log")
    axis.set_xlabel("Refinement iteration")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="major", color="0.82", linewidth=0.8)
    axis.grid(True, which="minor", color="0.92", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.xaxis.get_major_locator().set_params(integer=True)


def plot_history(
    dataframe: pd.DataFrame,
    csv_path: Path,
    include_relative_correction: bool = False,
) -> tuple[Figure, int]:
    """Create a two- or three-panel convergence-history figure."""
    factor = str(invariant_value(dataframe, "factor_precision", csv_path))
    boundary = factorization_boundary(factor)
    kappas = sorted(dataframe["requested_kappa"].unique())
    metrics = (
        ALL_METRICS if include_relative_correction else DEFAULT_METRICS
    )
    panel_count = len(metrics)
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.9, len(kappas))
    )
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")

    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(16.0 if panel_count == 3 else 11.2, 5.2),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    omitted_nonpositive = 0
    plotted_statuses: list[str] = []

    for index, kappa in enumerate(kappas):
        group = dataframe.loc[
            dataframe["requested_kappa"] == kappa
        ].sort_values("iteration")
        status = run_status(group)
        plotted_statuses.append(status)
        label = (
            f"{kappa_label(float(kappa), boundary)} "
            f"[{status_code(status)}]"
        )

        for axis, (column, _, _) in zip(axes, metrics):
            values = positive_log_values(group[column])
            omitted_nonpositive += int(
                (group[column].notna() & (group[column] <= 0.0)).sum()
            )

            axis.plot(
                group["iteration"],
                values,
                color=colors[index],
                marker=markers[index % len(markers)],
                markersize=4.2,
                linewidth=1.5,
                label=label,
            )

    for axis, (_, title, ylabel) in zip(axes, metrics):
        configure_axis(axis, title, ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    longest_label = max(map(len, labels), default=0)
    legend_columns = min(
        3 if longest_label > 28 else 4,
        len(labels),
    )
    legend_rows = int(np.ceil(len(labels) / legend_columns))
    bottom_margin = 0.18 + 0.055 * max(0, legend_rows - 2)

    figure.suptitle(figure_title(dataframe, csv_path), fontsize=14)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_columns,
        frameon=False,
        title=legend_title(boundary, plotted_statuses),
    )
    figure.tight_layout(rect=(0.0, bottom_margin, 1.0, 0.93))

    return figure, omitted_nonpositive


def main() -> int:
    """Plot all requested convergence-history datasets."""
    args = parse_arguments()
    raw_root, plots_root = resolve_results_roots(
        args.raw_root,
        args.plots_root,
    )
    csv_files = discover_csv_files(
        args.inputs,
        raw_root,
        DEFAULT_RAW_SUBDIRECTORY,
        CSV_PATTERN,
    )

    print(f"Found {len(csv_files)} convergence-history CSV file(s).")

    for csv_path in csv_files:
        dataframe = read_history(csv_path)
        figure, omitted_nonpositive = plot_history(
            dataframe,
            csv_path,
            include_relative_correction=args.include_relative_correction,
        )
        output_path = mirrored_plot_path(
            csv_path,
            raw_root,
            plots_root,
            args.format,
            DEFAULT_RAW_SUBDIRECTORY,
        )
        save_options: dict[str, object] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if args.format == "png":
            save_options["dpi"] = args.dpi

        figure.savefig(output_path, **save_options)
        plt.close(figure)

        statuses = ", ".join(sorted(dataframe["status"].unique()))
        print(
            f"Wrote {output_path} "
            f"({dataframe['requested_kappa'].nunique()} condition numbers; "
            f"statuses: {statuses})"
        )

        if omitted_nonpositive:
            warnings.warn(
                f"{csv_path.name}: omitted {omitted_nonpositive} "
                "nonpositive metric value(s) from logarithmic axes.",
                stacklevel=1,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())