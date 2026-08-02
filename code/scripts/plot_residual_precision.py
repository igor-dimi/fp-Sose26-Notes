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
from pathlib import Path
from typing import Iterable
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


CSV_PATTERN = "residual-precision__*.csv"
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

# Number of significand bits p. The convention used by condition_grids.hpp is
# u = 2**(-p), hence kappa_* = 1/u_f = 2**p.
PRECISION_SIGNIFICAND_BITS = {
    "fp8": 4,
    "bfloat16": 8,
    "fp16": 11,
    "fp32": 24,
    "fp64": 53,
    "fp128": 128,
    "fp256": 256,
}

CURVE_STYLES = {
    "fp64": {
        "label": "FP64 residual",
        "color": "#0072B2",
        "marker": "o",
    },
    "fp128": {
        "label": "FP128 residual",
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
    """Resolve raw- and plot-results roots."""
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
    """Resolve an explicit input against likely raw-results locations."""
    candidates = (
        path,
        raw_root / path,
        raw_root / "residual_precision" / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find input {path}. Checked:\n{checked}")


def discover_csv_files(inputs: Iterable[Path], raw_root: Path) -> list[Path]:
    """Discover the two residual-precision CSV files."""
    input_paths = list(inputs)
    if not input_paths:
        input_directory = raw_root / "residual_precision"
        if not input_directory.is_dir():
            raise FileNotFoundError(
                f"Default input directory does not exist: {input_directory}"
            )
        csv_files = list(input_directory.glob(CSV_PATTERN))
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
    if column not in dataframe.columns:
        raise ValueError(f"{path} is missing metadata column {column}.")
    values = dataframe[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"{path} must contain exactly one value for {column}; "
            f"found {len(values)}."
        )
    return values[0]


def read_sweep(csv_path: Path) -> pd.DataFrame:
    """Read and validate one residual-precision sweep."""
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

    for column in (
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
    ):
        one_metadata_value(dataframe, column, csv_path)

    residual_precision = str(
        one_metadata_value(dataframe, "residual_precision", csv_path)
    )
    if residual_precision not in EXPECTED_RESIDUAL_PRECISIONS:
        expected = ", ".join(EXPECTED_RESIDUAL_PRECISIONS)
        raise ValueError(
            f"{csv_path} has residual precision {residual_precision!r}; "
            f"expected one of: {expected}."
        )

    return dataframe.sort_values("requested_kappa").reset_index(drop=True)


def load_comparison(csv_files: Iterable[Path]) -> dict[str, pd.DataFrame]:
    """Load exactly one FP64- and one FP128-residual sweep."""
    sweeps: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}

    for csv_path in csv_files:
        dataframe = read_sweep(csv_path)
        residual = str(
            one_metadata_value(dataframe, "residual_precision", csv_path)
        )
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
                f"The requested_kappa grids do not match between "
                f"{reference_path} and {candidate_path}."
            )

        for column in MATCHED_METADATA_COLUMNS:
            if column not in reference.columns and column not in candidate.columns:
                continue
            reference_value = one_metadata_value(reference, column, reference_path)
            candidate_value = one_metadata_value(candidate, column, candidate_path)
            if reference_value != candidate_value:
                raise ValueError(
                    f"The sweeps differ in {column}: {reference_value!r} "
                    f"versus {candidate_value!r}."
                )

    factor = str(
        one_metadata_value(reference, "factor_precision", reference_path)
    )
    work = str(one_metadata_value(reference, "work_precision", reference_path))
    if factor != "fp32" or work != "fp64":
        raise ValueError(
            "Group C expects factor_precision='fp32' and "
            f"work_precision='fp64'; found {factor!r} and {work!r}."
        )

    return sweeps


def unit_roundoff(precision_name: str) -> float:
    """Return unit roundoff for a supported precision."""
    try:
        significand_bits = PRECISION_SIGNIFICAND_BITS[precision_name]
    except KeyError as error:
        supported = ", ".join(PRECISION_SIGNIFICAND_BITS)
        raise ValueError(
            f"Unknown precision {precision_name!r}; expected one of: {supported}."
        ) from error
    return math.ldexp(1.0, -significand_bits)


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
    parts = []
    if "converged" in counts:
        parts.append(f"{int(counts['converged'])} C")
    if "max-iterations" in counts:
        parts.append(f"{int(counts['max-iterations'])} M")
    for status, count in counts.items():
        if status not in ("converged", "max-iterations"):
            parts.append(f"{int(count)} {status}")
    return ", ".join(parts)


def plot_error_curve(
    axis: Axes,
    dataframe: pd.DataFrame,
    residual: str,
    y_column: str,
) -> None:
    """Plot one residual-precision error curve."""
    style = CURVE_STYLES[residual]
    label = f"{style['label']} ({status_counts(dataframe)})"
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


def make_figure(sweeps: dict[str, pd.DataFrame]) -> Figure:
    """Create the two-panel Group C comparison figure."""
    reference = sweeps[EXPECTED_RESIDUAL_PRECISIONS[0]]
    reference_path = Path("FP64 residual sweep")
    factor = str(
        one_metadata_value(reference, "factor_precision", reference_path)
    )
    work = str(one_metadata_value(reference, "work_precision", reference_path))
    measure = str(
        one_metadata_value(reference, "measure_precision", reference_path)
    )
    family = str(one_metadata_value(reference, "matrix_family", reference_path))
    dimension = int(one_metadata_value(reference, "dimension", reference_path))
    variant = str(one_metadata_value(reference, "variant", reference_path))

    factor_boundary = 1.0 / unit_roundoff(factor)
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
        "(a) Final normwise backward error, infinity norm", loc="left"
    )

    forward_axis.set_ylabel("Relative forward error")
    forward_axis.set_xlabel(r"Requested condition number $\kappa$")
    forward_axis.set_title(
        "(b) Final relative forward error, infinity norm", loc="left"
    )

    title = "Residual-precision comparison: FP32–FP64 working solve"
    subtitle = (
        f"{family.replace('-', ' ')}, n = {dimension}, {variant} residual, "
        f"measurement = {measure}"
    )
    figure.suptitle(title, fontsize=15, y=0.985)
    figure.text(0.5, 0.949, subtitle, ha="center", va="top", fontsize=10)

    boundary_text = format_number(factor_boundary)
    roundoff_text = format_number(work_roundoff)
    reference_text = (
        rf"Dashed vertical: $\kappa_*=1/u_f={boundary_text}$"
        rf"    Dotted horizontal: $u_{{\mathrm{{work}}}}={roundoff_text}$"
        rf"    Dash-dot (forward): reference slope $\kappa u_{{\mathrm{{work}}}}$"
    )
    figure.text(0.5, 0.919, reference_text, ha="center", va="top", fontsize=9)

    curve_handles, curve_labels = backward_axis.get_legend_handles_labels()
    reference_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            label="C = converged; M = maximum iterations",
        )
    ]
    figure.legend(
        handles=curve_handles + reference_handles,
        labels=curve_labels + ["C = converged; M = maximum iterations"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.55,
        columnspacing=1.3,
    )

    figure.subplots_adjust(top=0.82, bottom=0.10, left=0.12, right=0.98, hspace=0.28)
    return figure


def output_path_for(
    sweeps: dict[str, pd.DataFrame],
    plots_root: Path,
    output_format: str,
) -> Path:
    """Construct the deterministic comparison-figure path."""
    reference = sweeps[EXPECTED_RESIDUAL_PRECISIONS[0]]
    reference_path = Path("FP64 residual sweep")
    family = str(one_metadata_value(reference, "matrix_family", reference_path))
    dimension = int(one_metadata_value(reference, "dimension", reference_path))
    rhs = str(one_metadata_value(reference, "rhs_mode", reference_path))
    factor = str(
        one_metadata_value(reference, "factor_precision", reference_path)
    )
    work = str(one_metadata_value(reference, "work_precision", reference_path))
    measure = str(
        one_metadata_value(reference, "measure_precision", reference_path)
    )

    output_directory = plots_root / "residual_precision"
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
    raw_root, plots_root = resolve_roots(args)
    csv_files = discover_csv_files(args.inputs, raw_root)
    sweeps = load_comparison(csv_files)
    figure = make_figure(sweeps)
    output_path = output_path_for(sweeps, plots_root, args.format)
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