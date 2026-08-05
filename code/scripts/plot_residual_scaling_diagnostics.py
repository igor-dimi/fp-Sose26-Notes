#!/usr/bin/env python3
"""Plot Group E residual-conversion diagnostics for FP16 scaling.

By default, the script reads every residual-scaling CSV from
``results/raw/robustness/residual_scaling`` and writes one two-panel figure
per dataset to the corresponding directory under ``results/plots``.

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


CSV_PATTERN = "residual-scaling__*.csv"
DEFAULT_RAW_SUBDIRECTORY = Path("robustness/residual_scaling")
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
    "scale_residual",
    "record_residual_diagnostics",
    "status",
    "total_iterations",
    "iteration",
    "residual_inf_norm",
    "min_nonzero_abs",
    "nonzero_components",
    "zeroed_by_conversion",
}

NUMERIC_COLUMNS = (
    "dimension",
    "requested_kappa",
    "scale_residual",
    "record_residual_diagnostics",
    "total_iterations",
    "iteration",
    "residual_inf_norm",
    "min_nonzero_abs",
    "nonzero_components",
    "zeroed_by_conversion",
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
}

# IEEE binary16 has minimum positive subnormal 2**-24. Under
# round-to-nearest, values at or below the midpoint 2**-25 round to zero.
FP16_ROUND_TO_ZERO_THRESHOLD = 2.0**-25


def parse_arguments() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description=(
            "Plot FP16 residual-conversion diagnostics for scaled and "
            "unscaled iterative refinement."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Optional CSV files or directories. Directories are searched "
            f"recursively for {CSV_PATTERN}. If omitted, files under "
            "results/raw/robustness/residual_scaling are used."
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
    """Discover Group E residual-scaling CSV files."""
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


def read_diagnostics(csv_path: Path) -> pd.DataFrame:
    """Read, validate, and derive conversion-input diagnostics."""
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

    factor_precision = str(
        one_value(dataframe, "factor_precision", csv_path)
    ).lower()
    if factor_precision != "fp16":
        raise ValueError(
            f"{csv_path} uses factor precision {factor_precision!r}; "
            "this diagnostic plot expects fp16."
        )

    variants = set(dataframe["variant"].astype(str))
    if variants != set(EXPECTED_VARIANTS):
        expected = ", ".join(EXPECTED_VARIANTS)
        found = ", ".join(sorted(variants))
        raise ValueError(
            f"{csv_path} must contain variants {expected}; found {found}."
        )

    expected_scaling = {"unscaled": 0, "scaled": 1}
    expected_status = {"unscaled": "max-iterations", "scaled": "converged"}

    for variant in EXPECTED_VARIANTS:
        group = dataframe.loc[dataframe["variant"] == variant]
        scaling_flag = int(one_value(group, "scale_residual", csv_path))
        if scaling_flag != expected_scaling[variant]:
            raise ValueError(
                f"{csv_path}: variant {variant!r} has inconsistent "
                "scale_residual metadata."
            )

        if int(one_value(group, "record_residual_diagnostics", csv_path)) != 1:
            raise ValueError(
                f"{csv_path}: residual diagnostics were not enabled for "
                f"variant {variant!r}."
            )

        status = str(one_value(group, "status", csv_path))
        if status != expected_status[variant]:
            warnings.warn(
                f"{csv_path.name}: expected status {expected_status[variant]!r} "
                f"for {variant}, found {status!r}.",
                stacklevel=1,
            )

    integer_columns = (
        "dimension",
        "total_iterations",
        "iteration",
        "nonzero_components",
        "zeroed_by_conversion",
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

    if (dataframe["residual_inf_norm"] <= 0.0).any():
        raise ValueError(f"{csv_path} contains a nonpositive residual norm.")
    if (dataframe["min_nonzero_abs"] <= 0.0).any():
        raise ValueError(
            f"{csv_path} contains a nonpositive minimum residual component."
        )
    if (dataframe["nonzero_components"] <= 0).any():
        raise ValueError(f"{csv_path} contains no nonzero residual components.")
    if (
        dataframe["zeroed_by_conversion"]
        > dataframe["nonzero_components"]
    ).any():
        raise ValueError(
            f"{csv_path} reports more zeroed than nonzero components."
        )

    # The unscaled run converts r_k directly. The scaled run converts
    # r_k / ||r_k||_inf, so normalize its minimum component accordingly.
    dataframe["min_conversion_input_abs"] = dataframe["min_nonzero_abs"]
    scaled = dataframe["variant"] == "scaled"
    dataframe.loc[scaled, "min_conversion_input_abs"] = (
        dataframe.loc[scaled, "min_nonzero_abs"]
        / dataframe.loc[scaled, "residual_inf_norm"]
    )

    return dataframe.sort_values(["variant", "iteration"])


def configure_axis(axis: Axes, title: str, ylabel: str) -> None:
    """Apply common history-axis formatting."""
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
        f"Residual-conversion diagnostics: {precisions} "
        f"(n = {int(row['dimension'])}, "
        fr"$\kappa$ = {float(row['requested_kappa']):g})"
    )


def plot_diagnostics(dataframe: pd.DataFrame) -> Figure:
    """Create the two-panel residual-conversion diagnostic figure."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.0),
        sharex=False,
    )
    magnitude_axis, count_axis = axes

    for variant in EXPECTED_VARIANTS:
        group = dataframe.loc[
            dataframe["variant"] == variant
        ].sort_values("iteration")
        style = VARIANT_STYLES[variant]
        status = str(group["status"].iloc[0])
        code = STATUS_CODES.get(status, status)
        label = f"{style['label']} [{code}]"

        magnitude_axis.plot(
            group["iteration"],
            group["min_conversion_input_abs"],
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            linewidth=1.6,
            label=label,
        )
        count_axis.plot(
            group["iteration"],
            group["zeroed_by_conversion"],
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            linewidth=1.6,
            label=label,
        )

    magnitude_axis.axhline(
        FP16_ROUND_TO_ZERO_THRESHOLD,
        color="0.25",
        linestyle="--",
        linewidth=1.3,
        label=r"FP16 round-to-zero threshold $2^{-25}$",
    )
    magnitude_axis.set_yscale("log")
    configure_axis(
        magnitude_axis,
        "(a) Smallest conversion-input component",
        r"Minimum nonzero magnitude before FP16 conversion",
    )
    magnitude_axis.legend(frameon=False, loc="best")

    configure_axis(
        count_axis,
        "(b) Components lost during conversion",
        "Nonzero components rounded to zero",
    )
    count_axis.yaxis.get_major_locator().set_params(integer=True)
    max_zeroed = int(dataframe["zeroed_by_conversion"].max())
    count_axis.set_ylim(-0.5, max(1.0, max_zeroed + 1.5))
    count_axis.legend(frameon=False, loc="best")

    figure.suptitle(figure_title(dataframe), fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure


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
    """Save one diagnostic figure."""
    output_directory = output_directory_for(csv_path, raw_root, plots_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"{csv_path.stem}__diagnostics.{output_format}"
    )

    save_options: dict[str, object] = {"bbox_inches": "tight"}
    if output_format == "png":
        save_options["dpi"] = dpi

    figure.savefig(output_path, **save_options)
    plt.close(figure)
    return output_path


def main() -> None:
    """Plot all requested residual-scaling diagnostic datasets."""
    args = parse_arguments()
    raw_root, plots_root = resolve_roots(args)
    csv_files = discover_csv_files(args.inputs, raw_root)

    print(f"Found {len(csv_files)} residual-scaling CSV file(s).")

    for csv_path in csv_files:
        dataframe = read_diagnostics(csv_path)
        figure = plot_diagnostics(dataframe)
        output_path = save_plot(
            figure,
            csv_path,
            raw_root,
            plots_root,
            args.format,
            args.dpi,
        )

        unscaled = dataframe.loc[dataframe["variant"] == "unscaled"]
        scaled = dataframe.loc[dataframe["variant"] == "scaled"]
        print(
            f"Wrote {output_path} "
            f"(maximum components lost: unscaled="
            f"{int(unscaled['zeroed_by_conversion'].max())}, "
            f"scaled={int(scaled['zeroed_by_conversion'].max())})"
        )


if __name__ == "__main__":
    main()
