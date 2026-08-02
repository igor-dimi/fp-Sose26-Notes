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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


CSV_PATTERN = "condition-sweep__*.csv"

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


@dataclass(frozen=True)
class StatusStyle:
    """Visual and textual encoding for one algorithm termination status."""

    code: str
    description: str
    marker: str
    color: str


STATUS_STYLES = {
    "converged": StatusStyle("C", "converged", "o", "#0072B2"),
    "max-iterations": StatusStyle(
        "M", "maximum iterations", "^", "#E69F00"
    ),
    "diverged": StatusStyle("D", "diverged", "X", "#D55E00"),
    "non-finite": StatusStyle("N", "non-finite value", "P", "#CC79A7"),
    "factorization-input-non-finite": StatusStyle(
        "F", "non-finite factorization input", "s", "#000000"
    ),
}

# Number of significand bits p used by each supported precision. The boundary
# convention in condition_grids.hpp is kappa_* = 1/u_f = 2**p.
PRECISION_SIGNIFICAND_BITS = {
    "fp8": 4,
    "bfloat16": 8,
    "fp16": 11,
    "fp32": 24,
    "fp64": 53,
    "fp128": 128,
    "fp256": 256,
}


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
            "Root of the raw-results tree. Defaults to <code>/results/raw, "
            "where <code> is inferred from the script location."
        ),
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=None,
        help=(
            "Root of the plot-results tree. Defaults to "
            "<code>/results/plots."
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


def infer_code_directory() -> Path:
    """Infer the code directory for a script stored under ``code/scripts``."""
    script_path = Path(__file__).resolve()
    conventional_code_dir = script_path.parents[1]
    if (conventional_code_dir / "results" / "raw").is_dir():
        return conventional_code_dir

    current_directory = Path.cwd().resolve()
    if (current_directory / "results" / "raw").is_dir():
        return current_directory

    return conventional_code_dir


def resolve_roots(args: Namespace) -> tuple[Path, Path]:
    """Resolve the raw- and plot-results roots."""
    code_directory = infer_code_directory()
    raw_root = (
        args.raw_root.resolve()
        if args.raw_root is not None
        else code_directory / "results" / "raw"
    )
    plots_root = (
        args.plots_root.resolve()
        if args.plots_root is not None
        else code_directory / "results" / "plots"
    )
    return raw_root, plots_root


def resolve_input_path(path: Path, raw_root: Path) -> Path:
    """Resolve an explicit input against the current and raw directories."""
    candidates = (
        path,
        raw_root / path,
        raw_root / "condition_sweeps" / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find input {path}. Checked:\n{checked}")


def discover_csv_files(inputs: Iterable[Path], raw_root: Path) -> list[Path]:
    """Discover condition-sweep CSV files from explicit or default inputs."""
    input_paths = list(inputs)
    if not input_paths:
        sweep_directory = raw_root / "condition_sweeps"
        if not sweep_directory.is_dir():
            raise FileNotFoundError(
                f"Default input directory does not exist: {sweep_directory}"
            )
        csv_files = list(sweep_directory.glob(CSV_PATTERN))
    else:
        csv_files = []
        for input_path in input_paths:
            resolved_path = resolve_input_path(input_path, raw_root)
            if resolved_path.is_dir():
                csv_files.extend(resolved_path.rglob(CSV_PATTERN))
            elif resolved_path.suffix.lower() == ".csv":
                csv_files.append(resolved_path)
            else:
                raise ValueError(f"Input file is not a CSV file: {resolved_path}")

    unique_files = sorted({path.resolve() for path in csv_files})
    if not unique_files:
        raise FileNotFoundError(f"No files matching {CSV_PATTERN} were found.")
    return unique_files


def one_metadata_value(dataframe: pd.DataFrame, column: str, path: Path):
    """Return a metadata value after checking that it is constant."""
    values = dataframe[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"{path} must contain exactly one value for {column}; "
            f"found {len(values)}."
        )
    return values[0]


def read_sweep(csv_path: Path) -> pd.DataFrame:
    """Read and validate one condition-number sweep CSV file."""
    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    if dataframe.empty:
        raise ValueError(f"{csv_path} contains no data rows.")

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

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

    dataframe["status"] = dataframe["status"].astype(str).str.strip()
    unknown_statuses = sorted(set(dataframe["status"]) - set(STATUS_STYLES))
    if unknown_statuses:
        unknown = ", ".join(repr(status) for status in unknown_statuses)
        expected = ", ".join(STATUS_STYLES)
        raise ValueError(
            f"{csv_path} contains unknown status value(s): {unknown}. "
            f"Expected one of: {expected}."
        )

    for column in (
        "factor_precision",
        "work_precision",
        "residual_precision",
        "measure_precision",
        "matrix_family",
        "dimension",
        "variant",
        "max_iterations",
    ):
        one_metadata_value(dataframe, column, csv_path)

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


def unit_roundoff(precision_name: str) -> float:
    """Return the unit roundoff associated with a supported precision."""
    try:
        significand_bits = PRECISION_SIGNIFICAND_BITS[precision_name]
    except KeyError as error:
        supported = ", ".join(PRECISION_SIGNIFICAND_BITS)
        raise ValueError(
            f"Unknown precision {precision_name!r}; expected one of: {supported}."
        ) from error
    return math.ldexp(1.0, -significand_bits)


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
            edgecolors="white" if style.marker not in ("X", "P") else "none",
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
    factor = str(one_metadata_value(dataframe, "factor_precision", csv_path))
    work = str(one_metadata_value(dataframe, "work_precision", csv_path))
    residual = str(one_metadata_value(dataframe, "residual_precision", csv_path))
    measure = str(one_metadata_value(dataframe, "measure_precision", csv_path))
    family = str(one_metadata_value(dataframe, "matrix_family", csv_path))
    dimension = int(one_metadata_value(dataframe, "dimension", csv_path))
    variant = str(one_metadata_value(dataframe, "variant", csv_path))

    title = f"Condition-number sweep: {factor}–{work}–{residual}"
    subtitle = (
        f"{family.replace('-', ' ')}, n = {dimension}, {variant} residual, "
        f"measurement = {measure}"
    )
    return title, subtitle


def plot_sweep(dataframe: pd.DataFrame, csv_path: Path) -> Figure:
    """Create the two-panel Group B figure for one sweep."""
    factor = str(one_metadata_value(dataframe, "factor_precision", csv_path))
    work = str(one_metadata_value(dataframe, "work_precision", csv_path))
    max_iterations = int(one_metadata_value(dataframe, "max_iterations", csv_path))
    factor_boundary = 1.0 / unit_roundoff(factor)
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
    error_axis.set_title("(a) Final relative forward error, infinity norm", loc="left")

    iteration_axis.plot(
        kappas, iterations, color="#9a9a9a", linewidth=1.1, zorder=2
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

    figure.subplots_adjust(top=0.81, bottom=0.10, left=0.11, right=0.98, hspace=0.28)
    return figure


def output_path_for(
    csv_path: Path,
    raw_root: Path,
    plots_root: Path,
    output_format: str,
) -> Path:
    """Mirror a raw-results path beneath the plot-results root."""
    try:
        relative_path = csv_path.resolve().relative_to(raw_root.resolve())
        relative_parent = relative_path.parent
    except ValueError:
        relative_parent = Path("condition_sweeps")

    output_directory = plots_root / relative_parent
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / f"{csv_path.stem}.{output_format}"


def main() -> int:
    """Plot every selected condition-sweep CSV."""
    args = parse_arguments()
    raw_root, plots_root = resolve_roots(args)
    csv_files = discover_csv_files(args.inputs, raw_root)

    for csv_path in csv_files:
        dataframe = read_sweep(csv_path)
        figure = plot_sweep(dataframe, csv_path)
        output_path = output_path_for(
            csv_path,
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