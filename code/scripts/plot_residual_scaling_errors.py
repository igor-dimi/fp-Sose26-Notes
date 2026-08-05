#!/usr/bin/env python3
"""Plot Group E error histories with FP16 residual scaling on and off.

By default, the script reads every error-history CSV from
``results/raw/robustness/residual_scaling/error_histories`` and writes one
two-panel figure per dataset to the corresponding directory under
``results/plots``.

The intended location of this script is ``code/scripts``. Input files or
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


CSV_PATTERN = "residual-scaling__*__error-history.csv"
DEFAULT_RAW_SUBDIRECTORY = Path(
    "robustness/residual_scaling/error_histories"
)
EXPECTED_VARIANTS = ("unscaled", "scaled")

REQUIRED_COLUMNS = {
    "experiment",
    "matrix_family",
    "dimension",
    "factor_precision",
    "work_precision",
    "residual_precision",
    "measure_precision",
    "variant",
    "requested_kappa",
    "store_iterates",
    "scale_residual",
    "status",
    "total_iterations",
    "iteration",
    "forward_error_inf",
    "backward_error_inf",
}

NUMERIC_COLUMNS = (
    "dimension",
    "requested_kappa",
    "store_iterates",
    "scale_residual",
    "total_iterations",
    "iteration",
    "forward_error_inf",
    "backward_error_inf",
)

INVARIANT_METADATA = (
    "experiment",
    "matrix_family",
    "dimension",
    "factor_precision",
    "work_precision",
    "residual_precision",
    "measure_precision",
    "requested_kappa",
)

METRICS = (
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

VARIANT_STYLES = {
    "unscaled": {
        "label": "Unscaled",
        "color": "#D55E00",
        "marker": "s",
    },
    "scaled": {
        "label": "Scaled",
        "color": "#0072B2",
        "marker": "o",
    },
}

STATUS_CODES = {
    "converged": "C",
    "max-iterations": "M",
    "diverged": "D",
    "stagnated": "S",
    "non-finite": "N",
    "factorization-input-non-finite": "F",
    "factorization-failure": "F",
    "initial-solution-non-finite": "N",
    "residual-non-finite": "N",
    "correction-non-finite": "N",
    "iterate-non-finite": "N",
    "residual-conversion-underflow": "U",
}

STATUS_DESCRIPTIONS = {
    "C": "converged",
    "M": "maximum iterations",
    "D": "diverged",
    "S": "stagnated",
    "N": "non-finite value",
    "F": "factorization failure",
    "U": "residual-conversion underflow",
}

SIGNIFICAND_BITS = {
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
            "Plot forward- and backward-error histories for FP16 residual "
            "scaling enabled and disabled."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Optional CSV files or directories. Directories are searched "
            f"recursively for {CSV_PATTERN}. If omitted, files under "
            "results/raw/robustness/residual_scaling/error_histories are "
            "used."
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
    """Resolve an explicit input against likely raw-results locations."""
    candidates = (
        path,
        raw_root / path,
        raw_root / DEFAULT_RAW_SUBDIRECTORY / path,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find input {path}. Checked:\n{checked}")


def discover_csv_files(inputs: Iterable[Path], raw_root: Path) -> list[Path]:
    """Discover Group E error-history CSV files."""
    input_paths = list(inputs)

    if not input_paths:
        input_directory = raw_root / DEFAULT_RAW_SUBDIRECTORY
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


def one_value(dataframe: pd.DataFrame, column: str, csv_path: Path):
    """Return one invariant column value after validating it."""
    values = dataframe[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"{csv_path} must contain exactly one value for {column}; "
            f"found {len(values)}."
        )
    return values[0]


def read_error_history(csv_path: Path) -> pd.DataFrame:
    """Read and validate one scaled/unscaled error-history dataset."""
    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    if dataframe[list(NUMERIC_COLUMNS)].isna().any().any():
        bad_columns = dataframe[list(NUMERIC_COLUMNS)].columns[
            dataframe[list(NUMERIC_COLUMNS)].isna().any()
        ]
        raise ValueError(
            f"{csv_path} contains missing or invalid numeric data in: "
            + ", ".join(bad_columns)
        )

    for column in INVARIANT_METADATA:
        one_value(dataframe, column, csv_path)

    if one_value(dataframe, "experiment", csv_path) != "residual-scaling":
        raise ValueError(f"{csv_path} is not a residual-scaling experiment.")

    variants = set(dataframe["variant"].astype(str))
    if variants != set(EXPECTED_VARIANTS):
        expected = ", ".join(EXPECTED_VARIANTS)
        found = ", ".join(sorted(variants))
        raise ValueError(
            f"{csv_path} must contain variants {expected}; found {found}."
        )

    expected_scaling = {"unscaled": 0, "scaled": 1}
    for variant in EXPECTED_VARIANTS:
        group = dataframe.loc[dataframe["variant"] == variant]

        if int(one_value(group, "store_iterates", csv_path)) != 1:
            raise ValueError(
                f"{csv_path}: stored iterates were not enabled for "
                f"variant {variant!r}."
            )

        scaling_flag = int(one_value(group, "scale_residual", csv_path))
        if scaling_flag != expected_scaling[variant]:
            raise ValueError(
                f"{csv_path}: variant {variant!r} has inconsistent "
                "scale_residual metadata."
            )

        one_value(group, "status", csv_path)
        total_iterations = int(one_value(group, "total_iterations", csv_path))
        iterations = sorted(group["iteration"].astype(int))
        expected_iterations = list(range(total_iterations + 1))
        if iterations != expected_iterations:
            raise ValueError(
                f"{csv_path}: {variant!r} iterations do not equal "
                f"0,...,{total_iterations}."
            )

    integer_columns = (
        "dimension",
        "store_iterates",
        "scale_residual",
        "total_iterations",
        "iteration",
    )
    for column in integer_columns:
        if not np.allclose(dataframe[column], np.round(dataframe[column])):
            raise ValueError(f"{csv_path} contains nonintegral {column} values.")
        dataframe[column] = dataframe[column].astype(int)

    duplicates = dataframe.duplicated(
        subset=["variant", "iteration"], keep=False
    )
    if duplicates.any():
        raise ValueError(
            f"{csv_path} contains duplicate variant/iteration rows."
        )

    for column in ("forward_error_inf", "backward_error_inf"):
        values = dataframe[column]
        if (~np.isfinite(values)).any():
            raise ValueError(f"{csv_path} contains non-finite {column} values.")
        if (values < 0.0).any():
            raise ValueError(f"{csv_path} contains negative {column} values.")

    return dataframe.sort_values(["variant", "iteration"])


def unit_roundoff(precision: str) -> float | None:
    """Return 2**(-p) for a recognized floating-point precision."""
    significand_bits = SIGNIFICAND_BITS.get(precision.lower())
    if significand_bits is None:
        return None
    return float(2.0**-significand_bits)


def status_code(status: str) -> str:
    """Return the compact termination code for one run."""
    return STATUS_CODES.get(status, status)


def status_key(dataframe: pd.DataFrame) -> str:
    """Describe only status codes present in this dataset."""
    codes = {
        status_code(str(status))
        for status in dataframe["status"].unique()
    }
    definitions = (
        f"{code} = {description}"
        for code, description in STATUS_DESCRIPTIONS.items()
        if code in codes
    )
    return "Status: " + "; ".join(definitions)


def positive_log_values(values: pd.Series) -> pd.Series:
    """Replace zeros by NaN because they cannot appear on a log axis."""
    return values.where(values > 0.0)


def configure_axis(axis: Axes, title: str, ylabel: str) -> None:
    """Apply common history-axis formatting."""
    axis.set_yscale("log")
    axis.set_xlabel("Refinement iteration")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="major", color="0.82", linewidth=0.8)
    axis.grid(True, which="minor", color="0.92", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.xaxis.get_major_locator().set_params(integer=True)


def figure_title(dataframe: pd.DataFrame) -> str:
    """Construct a concise title from invariant metadata."""
    row = dataframe.iloc[0]
    precisions = (
        f"{row['factor_precision']} / {row['work_precision']} / "
        f"{row['residual_precision']}"
    )
    return (
        f"Effect of residual scaling: {precisions} "
        f"(n = {int(row['dimension'])}, "
        fr"$\kappa$ = {float(row['requested_kappa']):g})"
    )


def plot_error_history(dataframe: pd.DataFrame) -> tuple[Figure, int]:
    """Create the two-panel forward- and backward-error figure."""
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharex=False)
    omitted_zeros = 0

    for variant in EXPECTED_VARIANTS:
        group = dataframe.loc[
            dataframe["variant"] == variant
        ].sort_values("iteration")
        style = VARIANT_STYLES[variant]
        status = str(group["status"].iloc[0])
        label = f"{style['label']} [{status_code(status)}]"

        for axis, (column, _, _) in zip(axes, METRICS):
            values = positive_log_values(group[column])
            omitted_zeros += int((group[column] == 0.0).sum())
            axis.plot(
                group["iteration"],
                values,
                color=style["color"],
                marker=style["marker"],
                markersize=4.5,
                linewidth=1.6,
                label=label,
            )

    work_precision = str(dataframe.iloc[0]["work_precision"])
    work_roundoff = unit_roundoff(work_precision)

    for axis, (_, title, ylabel) in zip(axes, METRICS):
        configure_axis(axis, title, ylabel)
        if work_roundoff is not None:
            axis.axhline(
                work_roundoff,
                color="0.25",
                linestyle="--",
                linewidth=1.25,
                label=fr"$u_{{\mathrm{{{work_precision.upper()}}}}}$",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(figure_title(dataframe), fontsize=14)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(labels),
        frameon=False,
        title=status_key(dataframe),
    )
    figure.tight_layout(rect=(0.0, 0.18, 1.0, 0.93))
    return figure, omitted_zeros


def output_directory_for(
    csv_path: Path,
    raw_root: Path,
    plots_root: Path,
) -> Path:
    """Mirror the CSV's raw-results directory beneath the plots root."""
    try:
        relative_directory = csv_path.parent.relative_to(raw_root)
    except ValueError:
        relative_directory = DEFAULT_RAW_SUBDIRECTORY
    return plots_root / relative_directory


def save_plot(
    figure: Figure,
    csv_path: Path,
    raw_root: Path,
    plots_root: Path,
    output_format: str,
    dpi: int,
) -> Path:
    """Save one residual-scaling error-history figure."""
    output_directory = output_directory_for(csv_path, raw_root, plots_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"{csv_path.stem}__errors.{output_format}"
    )

    save_options: dict[str, object] = {"bbox_inches": "tight"}
    if output_format == "png":
        save_options["dpi"] = dpi

    figure.savefig(output_path, **save_options)
    plt.close(figure)
    return output_path


def main() -> None:
    """Plot all requested Group E error-history datasets."""
    args = parse_arguments()
    raw_root, plots_root = resolve_roots(args)
    csv_files = discover_csv_files(args.inputs, raw_root)

    print(f"Found {len(csv_files)} residual-scaling error-history CSV file(s).")

    for csv_path in csv_files:
        dataframe = read_error_history(csv_path)
        figure, omitted_zeros = plot_error_history(dataframe)
        output_path = save_plot(
            figure,
            csv_path,
            raw_root,
            plots_root,
            args.format,
            args.dpi,
        )

        statuses = ", ".join(
            f"{variant}={one_value(dataframe.loc[dataframe['variant'] == variant], 'status', csv_path)}"
            for variant in EXPECTED_VARIANTS
        )
        print(
            f"Wrote {output_path} "
            f"(kappa={float(dataframe['requested_kappa'].iloc[0]):g}; "
            f"{statuses})"
        )

        if omitted_zeros:
            warnings.warn(
                f"{csv_path.name}: omitted {omitted_zeros} zero error "
                "value(s) from logarithmic axes.",
                stacklevel=1,
            )


if __name__ == "__main__":
    main()
