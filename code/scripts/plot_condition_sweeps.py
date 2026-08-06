#!/usr/bin/env python3
"""Plot Group B mixed-precision condition-number sweeps.

By default, the script reads every condition-sweep CSV from
``results/raw/condition_sweeps`` and writes one two-panel figure per precision
configuration to ``results/plots/condition_sweeps``.

The intended location of this script is ``code/scripts``. Input files or
directories may also be supplied explicitly on the command line.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
import math
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

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
    unit_roundoff,
)
from mpir_plotting.styles import MIXED_IR_STATUS_NAMES, STATUS_STYLES


CSV_PATTERN = "condition-sweep__*.csv"
DEFAULT_RAW_SUBDIRECTORY = Path("condition_sweeps")

REQUIRED_COLUMNS = {
    "requested_kappa",
    "status",
    "total_iterations",
    "final_forward_error_inf",
    "factor_precision",
    "work_precision",
    "residual_precision",
    "measure_precision",
    "matrix_family",
    "dimension",
    "variant",
    "max_iterations",
}

NUMERIC_COLUMNS = (
    "requested_kappa",
    "total_iterations",
    "final_forward_error_inf",
    "max_iterations",
)

INVARIANT_METADATA_COLUMNS = (
    "factor_precision",
    "work_precision",
    "residual_precision",
    "measure_precision",
    "matrix_family",
    "dimension",
    "variant",
    "max_iterations",
)


def parse_arguments() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description=(
            "Plot Group B condition-number sweep CSV files as two-panel "
            "figures."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Optional CSV files or directories. Directories are searched "
            f"recursively for {CSV_PATTERN}. If omitted, all matching files "
            "under results/raw/condition_sweeps are plotted."
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

    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def read_sweep(csv_path: Path) -> pd.DataFrame:
    """Read and validate one condition-number sweep CSV file."""
    dataframe = read_csv_checked(csv_path, REQUIRED_COLUMNS)
    coerce_numeric_columns(dataframe, NUMERIC_COLUMNS, csv_path)

    if dataframe["requested_kappa"].isna().any():
        raise ValueError(f"{csv_path} contains an invalid requested_kappa.")
    if (dataframe["requested_kappa"] <= 0).any():
        raise ValueError(f"{csv_path} contains a non-positive requested_kappa.")
    if dataframe["requested_kappa"].duplicated().any():
        raise ValueError(f"{csv_path} contains duplicate requested_kappa rows.")

    if dataframe["total_iterations"].isna().any():
        raise ValueError(f"{csv_path} contains an invalid total_iterations.")
    if (dataframe["total_iterations"] < 0).any():
        raise ValueError(f"{csv_path} contains a negative total_iterations.")
    if not np.allclose(
        dataframe["total_iterations"],
        np.round(dataframe["total_iterations"]),
    ):
        raise ValueError(f"{csv_path} contains a non-integral total_iterations.")

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

    require_invariant_columns(
        dataframe,
        INVARIANT_METADATA_COLUMNS,
        csv_path,
    )

    invalid_errors = dataframe["final_forward_error_inf"].notna() & (
        ~np.isfinite(dataframe["final_forward_error_inf"])
        | (dataframe["final_forward_error_inf"] <= 0)
    )
    if invalid_errors.any():
        warnings.warn(
            f"{csv_path}: omitting {int(invalid_errors.sum())} non-positive "
            "or non-finite forward-error value(s) from the logarithmic panel."
        )
        dataframe.loc[invalid_errors, "final_forward_error_inf"] = np.nan

    return dataframe.sort_values("requested_kappa").reset_index(drop=True)


def format_number(value: float) -> str:
    """Format a positive value as a compact math-text fragment."""
    if (
        value >= 1.0
        and value < 1.0e9
        and math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9)
    ):
        return f"{round(value):,}".replace(",", "{,}")
    if 1.0e-3 <= value < 1.0e6:
        return f"{value:,.6g}"
    exponent = math.floor(math.log10(value))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.3g}\times 10^{{{exponent}}}"


def configure_axis(axis: Axes) -> None:
    """Apply the common log-x styling."""
    axis.set_xscale("log")
    axis.grid(True, which="major", color="#d8d8d8", linewidth=0.8)
    axis.grid(True, which="minor", color="#eeeeee", linewidth=0.5)
    axis.set_axisbelow(True)


def add_boundary_reference(
    axes: Iterable[Axes],
    boundary: float,
    minimum_kappa: float,
    maximum_kappa: float,
) -> bool:
    """Draw the theoretical boundary when it lies in the sampled range."""
    boundary_is_visible = minimum_kappa <= boundary <= maximum_kappa
    if boundary_is_visible:
        for axis in axes:
            axis.axvline(
                boundary,
                color="#333333",
                linestyle="--",
                linewidth=1.25,
                zorder=1,
            )
    return boundary_is_visible


def plot_status_points(
    axis: Axes,
    dataframe: pd.DataFrame,
    y_column: str,
) -> None:
    """Overlay status-coded markers for one plotted quantity."""
    for status, style in STATUS_STYLES.items():
        if status not in MIXED_IR_STATUS_NAMES:
            continue

        subset = dataframe[dataframe["status"] == status]
        subset = subset[np.isfinite(subset[y_column])]
        if subset.empty:
            continue

        axis.scatter(
            subset["requested_kappa"],
            subset[y_column],
            marker=style.marker,
            s=44,
            color=style.color,
            edgecolors="none" if style.marker == "X" else "white",
            linewidths=0.65,
            zorder=3,
        )


def make_status_handles(dataframe: pd.DataFrame) -> list[Line2D]:
    """Build legend handles for statuses present in the sweep."""
    handles: list[Line2D] = []
    counts = dataframe["status"].value_counts()

    for status, style in STATUS_STYLES.items():
        if status not in counts:
            continue

        count = int(counts[status])
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=style.marker,
                markersize=7,
                markerfacecolor=style.color,
                markeredgecolor=style.color,
                label=f"[{style.code}] {style.description} ({count})",
            )
        )

    return handles


def figure_title(dataframe: pd.DataFrame, csv_path: Path) -> tuple[str, str]:
    """Construct the title and metadata subtitle for one sweep."""
    factor = str(invariant_value(dataframe, "factor_precision", csv_path))
    work = str(invariant_value(dataframe, "work_precision", csv_path))
    residual = str(invariant_value(dataframe, "residual_precision", csv_path))
    measure = str(invariant_value(dataframe, "measure_precision", csv_path))
    family = str(invariant_value(dataframe, "matrix_family", csv_path))
    dimension = int(invariant_value(dataframe, "dimension", csv_path))
    variant = str(invariant_value(dataframe, "variant", csv_path))

    title = (
        "Condition-number sweep: "
        f"{precision_label(factor)}–{precision_label(work)}–"
        f"{precision_label(residual)}"
    )
    subtitle = (
        f"{family.replace('-', ' ')}, n = {dimension}, {variant} residual, "
        f"measurement = {precision_label(measure)}"
    )
    return title, subtitle


def plot_sweep(dataframe: pd.DataFrame, csv_path: Path) -> Figure:
    """Create the two-panel Group B figure for one sweep."""
    factor = str(invariant_value(dataframe, "factor_precision", csv_path))
    work = str(invariant_value(dataframe, "work_precision", csv_path))
    max_iterations = int(
        invariant_value(dataframe, "max_iterations", csv_path)
    )
    factor_boundary = factorization_boundary(factor)
    work_roundoff = unit_roundoff(work)

    kappas = dataframe["requested_kappa"].to_numpy(dtype=float)
    errors = dataframe["final_forward_error_inf"].to_numpy(dtype=float)
    iterations = dataframe["total_iterations"].to_numpy(dtype=float)

    figure, (error_axis, iteration_axis) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1.0)},
    )

    for axis in (error_axis, iteration_axis):
        configure_axis(axis)

    error_axis.set_yscale("log")
    error_axis.plot(kappas, errors, color="#9a9a9a", linewidth=1.1, zorder=2)
    plot_status_points(error_axis, dataframe, "final_forward_error_inf")
    error_axis.axhline(
        work_roundoff,
        color="#666666",
        linestyle=":",
        linewidth=1.1,
        zorder=1,
    )
    error_axis.set_ylabel("Relative forward error")
    error_axis.set_title(
        "(a) Final relative forward error, infinity norm",
        loc="left",
    )

    iteration_axis.plot(
        kappas,
        iterations,
        color="#9a9a9a",
        linewidth=1.1,
        zorder=2,
    )
    plot_status_points(iteration_axis, dataframe, "total_iterations")
    iteration_axis.axhline(
        max_iterations,
        color="#777777",
        linestyle=":",
        linewidth=1.0,
        zorder=1,
    )
    iteration_axis.set_ylim(-0.75, max_iterations + 1.25)
    iteration_axis.set_yticks(np.arange(0, max_iterations + 1, 5))
    iteration_axis.set_ylabel("Completed updates")
    iteration_axis.set_xlabel(r"Requested condition number $\kappa$")
    iteration_axis.set_title("(b) Iterative-refinement updates", loc="left")

    boundary_is_visible = add_boundary_reference(
        (error_axis, iteration_axis),
        factor_boundary,
        float(kappas.min()),
        float(kappas.max()),
    )

    title, subtitle = figure_title(dataframe, csv_path)
    figure.suptitle(title, fontsize=15, y=0.985)
    figure.text(0.5, 0.948, subtitle, ha="center", va="top", fontsize=10)

    boundary_text = format_number(factor_boundary)
    work_text = format_number(work_roundoff)
    if boundary_is_visible:
        reference_text = (
            rf"Dashed: $\kappa_*=1/u_f={boundary_text}$"
            rf"    Dotted (upper): $u_{{\mathrm{{work}}}}={work_text}$"
            "    Dotted (lower): update limit"
        )
    else:
        reference_text = (
            rf"$\kappa_*=1/u_f={boundary_text}$ lies outside the sweep"
            rf"    Dotted (upper): $u_{{\mathrm{{work}}}}={work_text}$"
            "    Dotted (lower): update limit"
        )
    figure.text(0.5, 0.915, reference_text, ha="center", va="top", fontsize=9)

    status_handles = make_status_handles(dataframe)
    figure.legend(
        handles=status_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=min(4, len(status_handles)),
        frameon=False,
        fontsize=9,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    missing_error_count = int(dataframe["final_forward_error_inf"].isna().sum())
    if missing_error_count:
        noun = "run has" if missing_error_count == 1 else "runs have"
        error_axis.text(
            0.015,
            0.965,
            f"{missing_error_count} {noun} no returned solution",
            transform=error_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#555555",
            bbox={
                "facecolor": "white",
                "edgecolor": "#dddddd",
                "boxstyle": "round,pad=0.25",
                "alpha": 0.9,
            },
        )

    figure.subplots_adjust(
        top=0.81,
        bottom=0.10,
        left=0.11,
        right=0.98,
        hspace=0.28,
    )
    return figure


def main() -> int:
    """Plot every selected condition-sweep CSV."""
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

    for csv_path in csv_files:
        dataframe = read_sweep(csv_path)
        figure = plot_sweep(dataframe, csv_path)
        output_path = mirrored_plot_path(
            csv_path,
            raw_root,
            plots_root,
            args.format,
            DEFAULT_RAW_SUBDIRECTORY,
        )
        figure.savefig(
            output_path,
            dpi=args.dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
