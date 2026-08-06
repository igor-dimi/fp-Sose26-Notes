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
from mpir_plotting.precisions import precision_label, unit_roundoff
from mpir_plotting.styles import (
    MIXED_IR_STATUS_NAMES,
    status_code,
    status_key,
)


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


def read_error_history(csv_path: Path) -> pd.DataFrame:
    """Read and validate one scaled/unscaled error-history dataset."""
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

    if int(invariant_value(dataframe, "dimension", csv_path)) <= 0:
        raise ValueError(f"{csv_path} contains a nonpositive dimension.")

    requested_kappa = float(
        invariant_value(dataframe, "requested_kappa", csv_path)
    )
    if not np.isfinite(requested_kappa) or requested_kappa <= 0.0:
        raise ValueError(f"{csv_path} contains an invalid requested_kappa.")

    variants = set(dataframe["variant"])
    if variants != set(EXPECTED_VARIANTS):
        expected = ", ".join(EXPECTED_VARIANTS)
        found = ", ".join(sorted(variants))
        raise ValueError(
            f"{csv_path} must contain variants {expected}; found {found}."
        )

    expected_scaling = {"unscaled": 0, "scaled": 1}
    for variant in EXPECTED_VARIANTS:
        group = dataframe.loc[dataframe["variant"] == variant]

        if int(invariant_value(group, "store_iterates", csv_path)) != 1:
            raise ValueError(
                f"{csv_path}: stored iterates were not enabled for "
                f"variant {variant!r}."
            )

        scaling_flag = int(
            invariant_value(group, "scale_residual", csv_path)
        )
        if scaling_flag != expected_scaling[variant]:
            raise ValueError(
                f"{csv_path}: variant {variant!r} has inconsistent "
                "scale_residual metadata."
            )

        invariant_value(group, "status", csv_path)
        total_iterations = int(
            invariant_value(group, "total_iterations", csv_path)
        )
        if total_iterations < 0:
            raise ValueError(
                f"{csv_path}: {variant!r} has negative total_iterations."
            )

        iterations = sorted(group["iteration"].tolist())
        expected_iterations = list(range(total_iterations + 1))
        if iterations != expected_iterations:
            raise ValueError(
                f"{csv_path}: {variant!r} iterations do not equal "
                f"0,...,{total_iterations}."
            )

    duplicates = dataframe.duplicated(
        subset=["variant", "iteration"],
        keep=False,
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

    return dataframe.sort_values(["variant", "iteration"]).reset_index(
        drop=True
    )


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
        f"Effect of residual scaling: {precisions} "
        f"(n = {dimension}, "
        fr"$\kappa$ = {requested_kappa:g})"
    )


def plot_error_history(
    dataframe: pd.DataFrame,
    csv_path: Path,
) -> tuple[Figure, int]:
    """Create the two-panel forward- and backward-error figure."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.0),
        sharex=False,
    )
    omitted_zeros = 0

    for variant in EXPECTED_VARIANTS:
        group = dataframe.loc[
            dataframe["variant"] == variant
        ].sort_values("iteration")
        style = VARIANT_STYLES[variant]
        status = str(invariant_value(group, "status", csv_path))
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

    work_precision = str(
        invariant_value(dataframe, "work_precision", csv_path)
    )
    work_roundoff = unit_roundoff(work_precision)
    work_label = precision_label(work_precision)

    for axis, (_, title, ylabel) in zip(axes, METRICS):
        configure_axis(axis, title, ylabel)
        axis.axhline(
            work_roundoff,
            color="0.25",
            linestyle="--",
            linewidth=1.25,
            label=fr"$u_{{\mathrm{{{work_label}}}}}$",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(figure_title(dataframe, csv_path), fontsize=14)
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(labels),
        frameon=False,
        title=status_key(dataframe["status"]),
    )
    figure.tight_layout(rect=(0.0, 0.18, 1.0, 0.93))
    return figure, omitted_zeros


def save_plot(
    figure: Figure,
    csv_path: Path,
    raw_root: Path,
    plots_root: Path,
    output_format: str,
    dpi: int,
) -> Path:
    """Save one residual-scaling error-history figure."""
    output_path = mirrored_plot_path(
        csv_path,
        raw_root,
        plots_root,
        output_format,
        DEFAULT_RAW_SUBDIRECTORY,
        suffix="__errors",
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

    print(f"Found {len(csv_files)} residual-scaling error-history CSV file(s).")

    for csv_path in csv_files:
        dataframe = read_error_history(csv_path)
        figure, omitted_zeros = plot_error_history(dataframe, csv_path)
        output_path = save_plot(
            figure,
            csv_path,
            raw_root,
            plots_root,
            args.format,
            args.dpi,
        )

        statuses = ", ".join(
            f"{variant}="
            f"{invariant_value(dataframe.loc[dataframe['variant'] == variant], 'status', csv_path)}"
            for variant in EXPECTED_VARIANTS
        )
        requested_kappa = float(
            invariant_value(dataframe, "requested_kappa", csv_path)
        )
        print(
            f"Wrote {output_path} "
            f"(kappa={requested_kappa:g}; {statuses})"
        )

        if omitted_zeros:
            warnings.warn(
                f"{csv_path.name}: omitted {omitted_zeros} zero error "
                "value(s) from logarithmic axes.",
                stacklevel=1,
            )


if __name__ == "__main__":
    main()