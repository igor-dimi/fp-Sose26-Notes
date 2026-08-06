#!/usr/bin/env python3
"""Plot Group E residual-conversion diagnostics for FP16 scaling.

By default, the script reads every residual-scaling comparison CSV from
``results/raw/robustness/residual_scaling`` and writes one two-panel figure
per dataset to the corresponding directory under ``results/plots``.

The intended location of this script is ``code/scripts``. Input files or
directories may also be supplied explicitly on the command line.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
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
from mpir_plotting.precisions import precision_label
from mpir_plotting.styles import MIXED_IR_STATUS_NAMES, status_code


CSV_PATTERN = "residual-scaling__*__comparison.csv"
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

INVARIANT_METADATA_COLUMNS = (
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


def read_diagnostics(csv_path: Path) -> pd.DataFrame:
    """Read, validate, and derive conversion-input diagnostics."""
    dataframe = read_csv_checked(csv_path, REQUIRED_COLUMNS)
    coerce_numeric_columns(
        dataframe,
        NUMERIC_COLUMNS,
        csv_path,
        require_complete=True,
    )

    dataframe["variant"] = (
        dataframe["variant"].astype(str).str.strip().str.lower()
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

    experiment = str(
        invariant_value(dataframe, "experiment", csv_path)
    ).strip().lower()
    if experiment != "residual-scaling":
        raise ValueError(f"{csv_path} is not a residual-scaling experiment.")

    factor_precision = str(
        invariant_value(dataframe, "factor_precision", csv_path)
    ).strip().lower()
    if factor_precision != "fp16":
        raise ValueError(
            f"{csv_path} uses factor precision {factor_precision!r}; "
            "this diagnostic plot expects fp16."
        )

    variants = set(dataframe["variant"])
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

        scaling_flag = int(
            invariant_value(group, "scale_residual", csv_path)
        )
        if scaling_flag != expected_scaling[variant]:
            raise ValueError(
                f"{csv_path}: variant {variant!r} has inconsistent "
                "scale_residual metadata."
            )

        diagnostics_flag = int(
            invariant_value(
                group,
                "record_residual_diagnostics",
                csv_path,
            )
        )
        if diagnostics_flag != 1:
            raise ValueError(
                f"{csv_path}: residual diagnostics were not enabled for "
                f"variant {variant!r}."
            )

        status = str(invariant_value(group, "status", csv_path))
        if status != expected_status[variant]:
            warnings.warn(
                f"{csv_path.name}: expected status "
                f"{expected_status[variant]!r} for {variant}, "
                f"found {status!r}.",
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
        subset=["variant", "iteration"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError(
            f"{csv_path} contains duplicate variant/iteration rows."
        )

    if (dataframe["requested_kappa"] <= 0.0).any():
        raise ValueError(f"{csv_path} contains a nonpositive requested_kappa.")
    if (dataframe["residual_inf_norm"] <= 0.0).any():
        raise ValueError(f"{csv_path} contains a nonpositive residual norm.")
    if (dataframe["min_nonzero_abs"] <= 0.0).any():
        raise ValueError(
            f"{csv_path} contains a nonpositive minimum residual component."
        )
    if (dataframe["nonzero_components"] <= 0).any():
        raise ValueError(f"{csv_path} contains no nonzero residual components.")
    if (dataframe["zeroed_by_conversion"] < 0).any():
        raise ValueError(
            f"{csv_path} contains a negative zeroed-component count."
        )
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

    return dataframe.sort_values(["variant", "iteration"]).reset_index(
        drop=True
    )


def configure_axis(axis: Axes, title: str, ylabel: str) -> None:
    """Apply common history-axis formatting."""
    axis.set_xlabel("Refinement iteration")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="major", color="0.82", linewidth=0.8)
    axis.grid(True, which="minor", color="0.92", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.xaxis.get_major_locator().set_params(integer=True)


def figure_title(dataframe: pd.DataFrame, csv_path: Path) -> str:
    """Construct a concise title from invariant metadata."""
    factor = str(
        invariant_value(dataframe, "factor_precision", csv_path)
    )
    work = str(invariant_value(dataframe, "work_precision", csv_path))
    residual = str(
        invariant_value(dataframe, "residual_precision", csv_path)
    )
    dimension = int(invariant_value(dataframe, "dimension", csv_path))
    requested_kappa = float(
        invariant_value(dataframe, "requested_kappa", csv_path)
    )

    precisions = " / ".join(
        precision_label(name) for name in (factor, work, residual)
    )
    return (
        f"Residual-conversion diagnostics: {precisions} "
        f"(n = {dimension}, "
        fr"$\kappa$ = {requested_kappa:g})"
    )


def plot_diagnostics(dataframe: pd.DataFrame, csv_path: Path) -> Figure:
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
        status = str(invariant_value(group, "status", csv_path))
        label = f"{style['label']} [{status_code(status)}]"

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
        "Minimum nonzero magnitude before FP16 conversion",
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

    figure.suptitle(figure_title(dataframe, csv_path), fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return figure


def save_plot(
    figure: Figure,
    csv_path: Path,
    raw_root: Path,
    plots_root: Path,
    output_format: str,
    dpi: int,
) -> Path:
    """Save one diagnostic figure."""
    output_path = mirrored_plot_path(
        csv_path,
        raw_root,
        plots_root,
        output_format,
        DEFAULT_RAW_SUBDIRECTORY,
        suffix="__diagnostics",
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

    print(f"Found {len(csv_files)} residual-scaling CSV file(s).")

    for csv_path in csv_files:
        dataframe = read_diagnostics(csv_path)
        figure = plot_diagnostics(dataframe, csv_path)
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