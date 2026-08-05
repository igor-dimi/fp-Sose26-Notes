#!/usr/bin/env python3
"""Plot Group A mixed-precision iterative-refinement histories.

By default, the script reads every convergence-history CSV from
``results/raw/convergence`` and writes one two-panel figure per precision
configuration to ``results/plots/convergence``.  Pass
``--include-relative-correction`` to add the relative-correction history as a
third panel.

The intended location of this script is ``code/scripts``.  Input files or
directories may also be supplied explicitly on the command line.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


CSV_PATTERN = "convergence-history__*.csv"

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

STATUS_CODES = {
    "converged": "C",
    "max-iterations": "M",
    "diverged": "D",
    "non-finite": "N",
    "factorization-input-non-finite": "F",
}

STATUS_DESCRIPTIONS = {
    "C": "converged",
    "M": "maximum iterations",
    "D": "diverged",
    "N": "non-finite value",
    "F": "non-finite factorization input",
}

# Number of significand bits p used by each current factorization type.
# The convergence-boundary convention in condition_grids.hpp is
# kappa_* = 1 / u_f = 2**p.
FACTOR_SIGNIFICAND_BITS = {
    "fp8": 4,
    "bfloat16": 8,
    "fp16": 11,
    "fp32": 24,
    "fp64": 53,
    "fp128": 128,
    "fp256": 256,
}

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
    """Resolve an explicit input against the current directory and raw tree."""
    candidates = (
        path,
        raw_root / path,
        raw_root / "convergence" / path,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find input {path}. Checked:\n{checked}"
    )


def discover_csv_files(inputs: Iterable[Path], raw_root: Path) -> list[Path]:
    """Discover convergence-history CSV files from explicit or default inputs."""
    input_paths = list(inputs)

    if not input_paths:
        convergence_directory = raw_root / "convergence"

        if not convergence_directory.is_dir():
            raise FileNotFoundError(
                "Default input directory does not exist: "
                f"{convergence_directory}"
            )

        csv_files = list(convergence_directory.glob(CSV_PATTERN))
    else:
        csv_files = []

        for input_path in input_paths:
            resolved_path = resolve_input_path(input_path, raw_root)

            if resolved_path.is_dir():
                csv_files.extend(resolved_path.rglob(CSV_PATTERN))
            elif resolved_path.is_file():
                if resolved_path.suffix.lower() != ".csv":
                    raise ValueError(
                        f"Input file is not a CSV file: {resolved_path}"
                    )
                csv_files.append(resolved_path)

    unique_files = sorted({path.resolve() for path in csv_files})

    if not unique_files:
        raise FileNotFoundError(
            f"No files matching {CSV_PATTERN} were found."
        )

    return unique_files


def read_history(csv_path: Path) -> pd.DataFrame:
    """Read and validate one convergence-history CSV file."""
    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}"
        )

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    if dataframe["requested_kappa"].isna().any():
        raise ValueError(
            f"{csv_path} contains a missing or invalid requested_kappa."
        )

    if dataframe["status"].isna().any():
        raise ValueError(f"{csv_path} contains a missing status.")

    dataframe["status"] = dataframe["status"].astype(str).str.strip()

    unknown_statuses = sorted(
        set(dataframe["status"].unique()) - set(STATUS_CODES)
    )

    if unknown_statuses:
        unknown = ", ".join(repr(status) for status in unknown_statuses)
        expected = ", ".join(STATUS_CODES)
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
        raise ValueError(
            f"{csv_path} contains no available iterates to plot."
        )

    nonintegral_iterations = (
        history["iteration"] % 1.0
    ).abs() > np.finfo(float).eps

    if nonintegral_iterations.any():
        raise ValueError(
            f"{csv_path} contains nonintegral iteration values."
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
        list(column for column, _, _ in ALL_METRICS)
    ].isna()

    if missing_metrics[["forward_error_inf", "backward_error_inf"]].any().any():
        raise ValueError(
            f"{csv_path} contains missing forward or backward errors for "
            "an available iterate."
        )

    if history.loc[history["iteration"] > 0, "rel_correction"].isna().any():
        raise ValueError(
            f"{csv_path} contains a missing relative correction after "
            "iteration 0."
        )

    metadata_columns = (
        "factor_precision",
        "work_precision",
        "residual_precision",
        "measure_precision",
        "matrix_family",
        "dimension",
        "variant",
    )

    varying_metadata = [
        column
        for column in metadata_columns
        if dataframe[column].nunique(dropna=False) != 1
    ]

    if varying_metadata:
        varying = ", ".join(varying_metadata)
        raise ValueError(
            f"{csv_path} combines incompatible metadata values: {varying}"
        )

    return history.sort_values(["requested_kappa", "iteration"])


def factorization_boundary(dataframe: pd.DataFrame) -> float | None:
    """Return kappa_* = 1 / u_f for a recognized factorization type."""
    factor_precision = str(dataframe.iloc[0]["factor_precision"]).lower()
    significand_bits = FACTOR_SIGNIFICAND_BITS.get(factor_precision)

    if significand_bits is None:
        warnings.warn(
            "No factorization-boundary metadata is known for precision "
            f"{factor_precision!r}; using plain kappa labels.",
            stacklevel=1,
        )
        return None

    boundary = float(2**significand_bits)

    if not np.isfinite(boundary):
        warnings.warn(
            f"The boundary for precision {factor_precision!r} exceeds "
            "the plotting range; using plain kappa labels.",
            stacklevel=1,
        )
        return None

    return boundary


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
    """Recognize a representative-grid point as a multiple of kappa_*."""
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


def kappa_label(kappa: float, boundary: float | None) -> str:
    """Format one condition number and expose its boundary-grid role."""
    value_text = number_math_text(kappa)

    if boundary is None:
        return fr"$\kappa={value_text}$"

    factor = representative_boundary_factor(kappa, boundary)

    if factor is not None:
        factor_text = boundary_factor_math_text(factor)
        return fr"${factor_text}={value_text}$"

    if np.isclose(kappa, 1.0, rtol=1.0e-12, atol=1.0e-15):
        return fr"$\kappa={value_text}$ (baseline)"

    return fr"$\kappa={value_text}$"


def status_code(group: pd.DataFrame) -> str:
    """Return the validated one-letter termination code for one run."""
    statuses = group["status"].unique()

    if len(statuses) != 1:
        raise ValueError(
            "A condition-number history must have exactly one status."
        )

    return STATUS_CODES[str(statuses[0])]


def status_key(codes: Iterable[str]) -> str:
    """Construct a compact explanation of the status codes in a figure."""
    present_codes = set(codes)
    definitions = (
        f"{code} = {description}"
        for code, description in STATUS_DESCRIPTIONS.items()
        if code in present_codes
    )
    return "Status: " + "; ".join(definitions)


def legend_title(boundary: float | None, codes: Iterable[str]) -> str:
    """Construct the boundary definition and status key for the legend."""
    lines: list[str] = []

    if boundary is not None:
        boundary_text = number_math_text(boundary)
        lines.append(fr"$\kappa_*=1/u_f={boundary_text}$")

    lines.append(status_key(codes))
    return "\n".join(lines)


def figure_title(dataframe: pd.DataFrame) -> str:
    """Construct a title from the invariant CSV metadata."""
    row = dataframe.iloc[0]
    precisions = (
        f"{row['factor_precision']} / {row['work_precision']} / "
        f"{row['residual_precision']}"
    )

    return (
        f"Convergence histories: {precisions} "
        f"(n = {int(row['dimension'])}, {row['variant']})"
    )


def positive_log_values(values: pd.Series) -> pd.Series:
    """Replace nonpositive values by NaN because they cannot appear on a log axis."""
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
    include_relative_correction: bool = False,
) -> tuple[Figure, int]:
    """Create a two- or three-panel convergence-history figure."""
    kappas = sorted(dataframe["requested_kappa"].unique())
    boundary = factorization_boundary(dataframe)
    metrics = (
        ALL_METRICS
        if include_relative_correction
        else DEFAULT_METRICS
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
    plotted_status_codes: list[str] = []

    for index, kappa in enumerate(kappas):
        group = dataframe.loc[
            dataframe["requested_kappa"] == kappa
        ].sort_values("iteration")
        code = status_code(group)
        plotted_status_codes.append(code)
        label = f"{kappa_label(float(kappa), boundary)} [{code}]"

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

    figure.suptitle(figure_title(dataframe), fontsize=14)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_columns,
        frameon=False,
        title=legend_title(boundary, plotted_status_codes),
    )
    figure.tight_layout(rect=(0.0, bottom_margin, 1.0, 0.93))

    return figure, omitted_nonpositive


def output_directory_for(
    csv_path: Path,
    raw_root: Path,
    plots_root: Path,
) -> Path:
    """Mirror the CSV's raw-results subdirectory beneath the plots root."""
    try:
        relative_directory = csv_path.parent.relative_to(raw_root)
    except ValueError:
        relative_directory = Path("convergence")

    return plots_root / relative_directory


def save_plot(
    figure: Figure,
    csv_path: Path,
    raw_root: Path,
    plots_root: Path,
    output_format: str,
    dpi: int,
) -> Path:
    """Save one figure to the mirrored plot-results directory."""
    output_directory = output_directory_for(
        csv_path,
        raw_root,
        plots_root,
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{csv_path.stem}.{output_format}"
    save_options: dict[str, object] = {
        "bbox_inches": "tight",
    }

    if output_format == "png":
        save_options["dpi"] = dpi

    figure.savefig(output_path, **save_options)
    plt.close(figure)

    return output_path


def main() -> None:
    """Plot all requested convergence-history datasets."""
    args = parse_arguments()
    raw_root, plots_root = resolve_roots(args)
    csv_files = discover_csv_files(args.inputs, raw_root)

    print(f"Found {len(csv_files)} convergence-history CSV file(s).")

    for csv_path in csv_files:
        dataframe = read_history(csv_path)
        figure, omitted_nonpositive = plot_history(
            dataframe,
            include_relative_correction=args.include_relative_correction,
        )
        output_path = save_plot(
            figure,
            csv_path,
            raw_root,
            plots_root,
            args.format,
            args.dpi,
        )

        statuses = ", ".join(
            sorted(dataframe["status"].dropna().unique())
        )
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


if __name__ == "__main__":
    main()