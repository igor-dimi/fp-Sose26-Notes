#!/usr/bin/env python3
"""Plot the Group C residual-precision comparison.

By default, the script reads the FP32--FP64--FP64 and FP32--FP64--FP128
residual-precision sweeps from ``results/raw/residual_precision`` and writes a
single two-panel comparison figure to ``results/plots/residual_precision``.

The intended location of this script is ``code/scripts``. The two CSV files or
a directory containing them may also be supplied explicitly.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
import math
from pathlib import Path

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
    mirrored_plot_directory,
    resolve_results_roots,
)
from mpir_plotting.precisions import (
    factorization_boundary,
    precision_label,
    unit_roundoff,
)
from mpir_plotting.styles import (
    MIXED_IR_STATUS_NAMES,
    STATUS_STYLES,
    status_code,
    status_key,
)


CSV_PATTERN = "residual-precision__*.csv"
DEFAULT_RAW_SUBDIRECTORY = Path("residual_precision")
EXPECTED_RESIDUAL_PRECISIONS = ("fp64", "fp128")

REQUIRED_COLUMNS = {
    "experiment",
    "matrix_family",
    "dimension",
    "factor_precision",
    "work_precision",
    "residual_precision",
    "measure_precision",
    "variant",
    "rhs_mode",
    "requested_kappa",
    "max_iterations",
    "status",
    "total_iterations",
    "final_forward_error_inf",
    "final_backward_error_inf",
}

NUMERIC_COLUMNS = (
    "requested_kappa",
    "total_iterations",
    "final_forward_error_inf",
    "final_backward_error_inf",
    "max_iterations",
)

INVARIANT_METADATA_COLUMNS = (
    "experiment",
    "matrix_family",
    "dimension",
    "factor_precision",
    "work_precision",
    "residual_precision",
    "measure_precision",
    "variant",
    "rhs_mode",
    "max_iterations",
)

MATCHED_METADATA_COLUMNS = (
    "experiment",
    "matrix_family",
    "dimension",
    "factor_precision",
    "work_precision",
    "measure_precision",
    "variant",
    "rhs_mode",
    "matrix_seed_u",
    "matrix_seed_v",
    "vector_seed",
    "rotation_theta",
    "max_iterations",
    "effective_rel_correction_tol",
    "detect_divergence",
    "divergence_growth_factor",
    "divergence_growth_steps",
    "scale_residual",
)

CURVE_STYLES = {
    "fp64": {
        "color": "#0072B2",
        "marker": "o",
    },
    "fp128": {
        "color": "#D55E00",
        "marker": "s",
    },
}


def parse_arguments() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description=(
            "Compare Group C FP64 and FP128 residual-precision sweeps in "
            "forward- and backward-error log-log panels."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Optional CSV files or directories. Directories are searched "
            f"recursively for {CSV_PATTERN}. If omitted, files under "
            "results/raw/residual_precision are used."
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
    """Read and validate one residual-precision sweep."""
    dataframe = read_csv_checked(csv_path, REQUIRED_COLUMNS)
    coerce_numeric_columns(dataframe, NUMERIC_COLUMNS, csv_path)

    if dataframe["requested_kappa"].isna().any():
        raise ValueError(f"{csv_path} contains an invalid requested_kappa.")
    if (dataframe["requested_kappa"] <= 0).any():
        raise ValueError(f"{csv_path} contains a non-positive requested_kappa.")
    if dataframe["requested_kappa"].duplicated().any():
        raise ValueError(f"{csv_path} contains duplicate requested_kappa rows.")

    for column in ("final_forward_error_inf", "final_backward_error_inf"):
        invalid = (
            dataframe[column].isna()
            | ~np.isfinite(dataframe[column])
            | (dataframe[column] <= 0)
        )
        if invalid.any():
            raise ValueError(
                f"{csv_path} contains {int(invalid.sum())} unavailable, "
                f"non-positive, or non-finite {column} value(s)."
            )

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

    residual_precision = str(
        invariant_value(dataframe, "residual_precision", csv_path)
    ).strip().lower()
    if residual_precision not in EXPECTED_RESIDUAL_PRECISIONS:
        expected = ", ".join(EXPECTED_RESIDUAL_PRECISIONS)
        raise ValueError(
            f"{csv_path} has residual precision {residual_precision!r}; "
            f"expected one of: {expected}."
        )

    return dataframe.sort_values("requested_kappa").reset_index(drop=True)


def load_comparison(
    csv_files: Iterable[Path],
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Load exactly one FP64- and one FP128-residual sweep."""
    sweeps: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}

    for csv_path in csv_files:
        dataframe = read_sweep(csv_path)
        residual = str(
            invariant_value(dataframe, "residual_precision", csv_path)
        ).strip().lower()
        if residual in sweeps:
            raise ValueError(
                f"Multiple {residual} residual sweeps were supplied: "
                f"{paths[residual]} and {csv_path}."
            )
        sweeps[residual] = dataframe
        paths[residual] = csv_path

    missing = set(EXPECTED_RESIDUAL_PRECISIONS) - set(sweeps)
    if missing:
        raise ValueError(
            "The comparison requires one sweep for each residual precision; "
            f"missing: {', '.join(sorted(missing))}."
        )

    reference_residual = EXPECTED_RESIDUAL_PRECISIONS[0]
    reference = sweeps[reference_residual]
    reference_path = paths[reference_residual]

    for residual in EXPECTED_RESIDUAL_PRECISIONS[1:]:
        candidate = sweeps[residual]
        candidate_path = paths[residual]
        if not np.array_equal(
            reference["requested_kappa"].to_numpy(),
            candidate["requested_kappa"].to_numpy(),
        ):
            raise ValueError(
                "The requested_kappa grids do not match between "
                f"{reference_path} and {candidate_path}."
            )

        for column in MATCHED_METADATA_COLUMNS:
            if column not in reference.columns and column not in candidate.columns:
                continue
            reference_value = invariant_value(
                reference,
                column,
                reference_path,
            )
            candidate_value = invariant_value(
                candidate,
                column,
                candidate_path,
            )
            if reference_value != candidate_value:
                raise ValueError(
                    f"The sweeps differ in {column}: {reference_value!r} "
                    f"versus {candidate_value!r}."
                )

    factor = str(
        invariant_value(reference, "factor_precision", reference_path)
    ).strip().lower()
    work = str(
        invariant_value(reference, "work_precision", reference_path)
    ).strip().lower()
    if factor != "fp32" or work != "fp64":
        raise ValueError(
            "Group C expects factor_precision='fp32' and "
            f"work_precision='fp64'; found {factor!r} and {work!r}."
        )

    return sweeps, paths


def format_number(value: float) -> str:
    """Format a positive number for a math-text annotation."""
    if (
        1.0 <= value < 1.0e9
        and math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9)
    ):
        return f"{round(value):,}".replace(",", "{,}")
    exponent = math.floor(math.log10(value))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.3g}\times 10^{{{exponent}}}"


def configure_loglog_axis(axis: Axes) -> None:
    """Apply common log-log styling."""
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="major", color="#d8d8d8", linewidth=0.8)
    axis.grid(True, which="minor", color="#eeeeee", linewidth=0.5)
    axis.set_axisbelow(True)


def status_counts(dataframe: pd.DataFrame) -> str:
    """Return compact termination-status counts for a legend label."""
    counts = dataframe["status"].value_counts()
    parts = [
        f"{int(counts[status])} {status_code(status)}"
        for status in STATUS_STYLES
        if status in counts
    ]
    return ", ".join(parts)


def plot_error_curve(
    axis: Axes,
    dataframe: pd.DataFrame,
    residual: str,
    y_column: str,
) -> None:
    """Plot one residual-precision error curve."""
    style = CURVE_STYLES[residual]
    label = (
        f"{precision_label(residual)} residual "
        f"({status_counts(dataframe)})"
    )
    axis.plot(
        dataframe["requested_kappa"],
        dataframe[y_column],
        color=style["color"],
        marker=style["marker"],
        markersize=4.1,
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.25,
        label=label,
        zorder=3,
    )


def make_figure(
    sweeps: dict[str, pd.DataFrame],
    paths: dict[str, Path],
) -> Figure:
    """Create the two-panel Group C comparison figure."""
    reference_residual = EXPECTED_RESIDUAL_PRECISIONS[0]
    reference = sweeps[reference_residual]
    reference_path = paths[reference_residual]

    factor = str(
        invariant_value(reference, "factor_precision", reference_path)
    ).strip().lower()
    work = str(
        invariant_value(reference, "work_precision", reference_path)
    ).strip().lower()
    measure = str(
        invariant_value(reference, "measure_precision", reference_path)
    ).strip().lower()
    family = str(
        invariant_value(reference, "matrix_family", reference_path)
    )
    dimension = int(invariant_value(reference, "dimension", reference_path))
    variant = str(invariant_value(reference, "variant", reference_path))

    factor_boundary = factorization_boundary(factor)
    work_roundoff = unit_roundoff(work)
    kappas = reference["requested_kappa"].to_numpy(dtype=float)

    figure, (backward_axis, forward_axis) = plt.subplots(
        2,
        1,
        figsize=(10.8, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.12)},
    )

    for axis in (backward_axis, forward_axis):
        configure_loglog_axis(axis)
        axis.axvline(
            factor_boundary,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )
        axis.axhline(
            work_roundoff,
            color="#666666",
            linestyle=":",
            linewidth=1.1,
            zorder=1,
        )

    for residual in EXPECTED_RESIDUAL_PRECISIONS:
        plot_error_curve(
            backward_axis,
            sweeps[residual],
            residual,
            "final_backward_error_inf",
        )
        plot_error_curve(
            forward_axis,
            sweeps[residual],
            residual,
            "final_forward_error_inf",
        )

    # Table 7.1 predicts a conditioning-dependent forward-error floor when
    # u_r = u. This line is a slope reference, not a pointwise error bound.
    forward_axis.plot(
        kappas,
        kappas * work_roundoff,
        color="#777777",
        linestyle="-.",
        linewidth=1.05,
        zorder=2,
    )

    backward_axis.set_ylabel("Relative backward error")
    backward_axis.set_title(
        "(a) Final normwise backward error, infinity norm",
        loc="left",
    )

    forward_axis.set_ylabel("Relative forward error")
    forward_axis.set_xlabel(r"Requested condition number $\kappa$")
    forward_axis.set_title(
        "(b) Final relative forward error, infinity norm",
        loc="left",
    )

    title = (
        "Residual-precision comparison: "
        f"{precision_label(factor)}–{precision_label(work)} working solve"
    )
    subtitle = (
        f"{family.replace('-', ' ')}, n = {dimension}, {variant} residual, "
        f"measurement = {precision_label(measure)}"
    )
    figure.suptitle(title, fontsize=15, y=0.985)
    figure.text(0.5, 0.949, subtitle, ha="center", va="top", fontsize=10)

    boundary_text = format_number(factor_boundary)
    roundoff_text = format_number(work_roundoff)
    reference_text = (
        rf"Dashed vertical: $\kappa_*=1/u_f={boundary_text}$"
        rf"    Dotted horizontal: $u_{{\mathrm{{work}}}}={roundoff_text}$"
        rf"    Dash-dot (forward): reference slope "
        rf"$\kappa u_{{\mathrm{{work}}}}$"
    )
    figure.text(0.5, 0.919, reference_text, ha="center", va="top", fontsize=9)

    curve_handles, curve_labels = backward_axis.get_legend_handles_labels()
    all_statuses = (
        status
        for residual in EXPECTED_RESIDUAL_PRECISIONS
        for status in sweeps[residual]["status"]
    )
    status_definition = status_key(all_statuses).removeprefix("Status: ")
    status_handle = Line2D([], [], linestyle="none", label=status_definition)
    figure.legend(
        handles=curve_handles + [status_handle],
        labels=curve_labels + [status_definition],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.55,
        columnspacing=1.3,
    )

    figure.subplots_adjust(
        top=0.82,
        bottom=0.10,
        left=0.12,
        right=0.98,
        hspace=0.28,
    )
    return figure


def output_path_for(
    sweeps: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    raw_root: Path,
    plots_root: Path,
    output_format: str,
) -> Path:
    """Construct the deterministic comparison-figure path."""
    reference_residual = EXPECTED_RESIDUAL_PRECISIONS[0]
    reference = sweeps[reference_residual]
    reference_path = paths[reference_residual]

    family = str(
        invariant_value(reference, "matrix_family", reference_path)
    )
    dimension = int(invariant_value(reference, "dimension", reference_path))
    rhs = str(invariant_value(reference, "rhs_mode", reference_path))
    factor = str(
        invariant_value(reference, "factor_precision", reference_path)
    ).strip().lower()
    work = str(
        invariant_value(reference, "work_precision", reference_path)
    ).strip().lower()
    measure = str(
        invariant_value(reference, "measure_precision", reference_path)
    ).strip().lower()

    output_directory = mirrored_plot_directory(
        reference_path,
        raw_root,
        plots_root,
        DEFAULT_RAW_SUBDIRECTORY,
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    filename = (
        f"residual-precision__{family}__n-{dimension}__{rhs}__"
        f"uf-{factor}__u-{work}__ur-fp64-vs-fp128__um-{measure}."
        f"{output_format}"
    )
    return output_directory / filename


def main() -> int:
    """Load the two sweeps and write their comparison figure."""
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
    sweeps, paths = load_comparison(csv_files)
    figure = make_figure(sweeps, paths)
    output_path = output_path_for(
        sweeps,
        paths,
        raw_root,
        plots_root,
        args.format,
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
