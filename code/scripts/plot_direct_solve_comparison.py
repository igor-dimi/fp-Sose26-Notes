#!/usr/bin/env python3
"""Plot iterative-refinement/direct-LU condition-sweep comparisons.

By default, the script reads every direct-solve-comparison CSV from
``results/raw/direct_comparison`` and writes one two-panel log-log figure per
dataset to ``results/plots/direct_comparison``.

Precision roles are read from each CSV.  A directory may therefore contain,
for example, both FP32-FP64-FP128 and FP64-FP64-FP64 iterative-refinement
runs; each dataset receives its own labels and reference lines.

The intended location of this script is ``code/scripts``. Input files or
directories may also be supplied explicitly on the command line.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
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

from mpir_plotting.csv_validation import (
    coerce_numeric_columns,
    invariant_value,
    read_csv_checked,
)
from mpir_plotting.paths import (
    discover_csv_files,
    mirrored_plot_path,
    resolve_results_roots,
)
from mpir_plotting.precisions import (
    PRECISIONS,
    factorization_boundary,
    is_supported_precision,
    precision_label,
    unit_roundoff,
)
from mpir_plotting.styles import (
    DIRECT_SOLVE_STATUS_NAMES,
    MIXED_IR_STATUS_NAMES,
    STATUS_STYLES,
)


CSV_PATTERN = "direct-solve-comparison__*.csv"
DEFAULT_RAW_SUBDIRECTORY = Path("direct_comparison")
EXPECTED_VARIANTS = ("mixed-ir", "direct-lu")

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
)

COMMON_METADATA_COLUMNS = (
    "experiment",
    "matrix_family",
    "dimension",
    "measure_precision",
    "rhs_mode",
    "matrix_seed_u",
    "matrix_seed_v",
    "vector_seed",
    "rotation_theta",
)

METHOD_STYLES = {
    "mixed-ir": {
        "color": "#0072B2",
    },
    "direct-lu": {
        "color": "#D55E00",
    },
}


@dataclass(frozen=True)
class PrecisionRoles:
    """Precision metadata for one method in a comparison dataset."""

    factor: str
    work: str
    residual: str
    measure: str

def parse_arguments() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description=(
            "Compare iterative refinement with a direct LU solve in "
            "forward- and backward-error panels. Precision labels are "
            "derived from each CSV."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Optional CSV files or directories. Directories are searched "
            f"recursively for {CSV_PATTERN}. If omitted, files under "
            "results/raw/direct_comparison are used."
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


def validate_precision_roles(
    sweeps: dict[str, pd.DataFrame],
    csv_path: Path,
) -> dict[str, PrecisionRoles]:
    """Validate and return the precision roles recorded by both methods.

    The iterative-refinement triple is intentionally unrestricted.  The
    direct baseline must use one arithmetic precision for factorization and
    solution and must mark residual precision as ``none``.
    """

    roles: dict[str, PrecisionRoles] = {}
    for variant, subset in sweeps.items():
        roles[variant] = PrecisionRoles(
            factor=str(
                invariant_value(subset, "factor_precision", csv_path)
            ).strip().lower(),
            work=str(
                invariant_value(subset, "work_precision", csv_path)
            ).strip().lower(),
            residual=str(
                invariant_value(subset, "residual_precision", csv_path)
            ).strip().lower(),
            measure=str(
                invariant_value(subset, "measure_precision", csv_path)
            ).strip().lower(),
        )

    iterative = roles["mixed-ir"]
    direct = roles["direct-lu"]

    for role_name in ("factor", "work", "residual", "measure"):
        precision_name = getattr(iterative, role_name)
        if not is_supported_precision(precision_name):
            supported = ", ".join(PRECISIONS)
            raise ValueError(
                f"{csv_path}: iterative-refinement {role_name} precision "
                f"{precision_name!r} is unknown; expected one of: "
                f"{supported}."
            )

    if direct.factor != direct.work:
        raise ValueError(
            f"{csv_path}: direct LU must factor and solve in one precision; "
            f"found factor={direct.factor!r}, work={direct.work!r}."
        )
    if direct.residual != "none":
        raise ValueError(
            f"{csv_path}: direct LU residual precision must be 'none'; "
            f"found {direct.residual!r}."
        )
    for role_name in ("factor", "work", "measure"):
        precision_name = getattr(direct, role_name)
        if not is_supported_precision(precision_name):
            supported = ", ".join(PRECISIONS)
            raise ValueError(
                f"{csv_path}: direct-LU {role_name} precision "
                f"{precision_name!r} is unknown; expected one of: "
                f"{supported}."
            )
    if iterative.measure != direct.measure:
        raise ValueError(
            f"{csv_path}: both methods must use the same measurement "
            f"precision; found {iterative.measure!r} and "
            f"{direct.measure!r}."
        )

    return roles


def read_comparison(csv_path: Path) -> dict[str, pd.DataFrame]:
    """Read and validate one Group D comparison CSV."""
    dataframe = read_csv_checked(csv_path, REQUIRED_COLUMNS)
    coerce_numeric_columns(dataframe, NUMERIC_COLUMNS, csv_path)

    if dataframe["requested_kappa"].isna().any():
        raise ValueError(f"{csv_path} contains an invalid requested_kappa.")
    if (dataframe["requested_kappa"] <= 0).any():
        raise ValueError(f"{csv_path} contains a non-positive requested_kappa.")

    dataframe["variant"] = (
        dataframe["variant"].astype(str).str.strip().str.lower()
    )
    dataframe["status"] = (
        dataframe["status"].astype(str).str.strip().str.lower()
    )
    actual_variants = set(dataframe["variant"])
    if actual_variants != set(EXPECTED_VARIANTS):
        raise ValueError(
            f"{csv_path} must contain variants {EXPECTED_VARIANTS}; "
            f"found {tuple(sorted(actual_variants))}."
        )

    duplicates = dataframe.duplicated(
        subset=["requested_kappa", "variant"], keep=False
    )
    if duplicates.any():
        raise ValueError(
            f"{csv_path} contains duplicate rows for a condition number "
            "and variant."
        )

    counts = dataframe.groupby("requested_kappa")["variant"].nunique()
    if (counts != len(EXPECTED_VARIANTS)).any():
        raise ValueError(
            f"{csv_path} does not contain both variants at every condition "
            "number."
        )

    for column in COMMON_METADATA_COLUMNS:
        if column in dataframe.columns:
            invariant_value(dataframe, column, csv_path)

    if str(invariant_value(dataframe, "experiment", csv_path)) != (
        "direct-solve-comparison"
    ):
        raise ValueError(
            f"{csv_path} is not a direct-solve-comparison experiment."
        )

    for column in ("final_forward_error_inf", "final_backward_error_inf"):
        invalid = dataframe[column].notna() & (
            ~np.isfinite(dataframe[column]) | (dataframe[column] <= 0)
        )
        if invalid.any():
            warnings.warn(
                f"{csv_path.name}: omitting {int(invalid.sum())} non-positive "
                f"or non-finite {column} value(s).",
                stacklevel=1,
            )
            dataframe.loc[invalid, column] = np.nan

    sweeps = {
        variant: dataframe[dataframe["variant"] == variant]
        .sort_values("requested_kappa")
        .reset_index(drop=True)
        for variant in EXPECTED_VARIANTS
    }

    reference_kappas = sweeps[EXPECTED_VARIANTS[0]][
        "requested_kappa"
    ].to_numpy()
    for variant in EXPECTED_VARIANTS[1:]:
        if not np.array_equal(
            reference_kappas,
            sweeps[variant]["requested_kappa"].to_numpy(),
        ):
            raise ValueError(
                f"{csv_path}: requested_kappa grids do not match between "
                "the two variants."
            )

    unknown_mixed_statuses = sorted(
        set(sweeps["mixed-ir"]["status"]) - MIXED_IR_STATUS_NAMES
    )
    if unknown_mixed_statuses:
        raise ValueError(
            f"{csv_path} contains unknown mixed-IR status value(s): "
            f"{', '.join(map(repr, unknown_mixed_statuses))}."
        )

    unknown_direct_statuses = sorted(
        set(sweeps["direct-lu"]["status"]) - DIRECT_SOLVE_STATUS_NAMES
    )
    if unknown_direct_statuses:
        raise ValueError(
            f"{csv_path} contains unknown direct-LU status value(s): "
            f"{', '.join(map(repr, unknown_direct_statuses))}."
        )

    validate_precision_roles(sweeps, csv_path)
    return sweeps


def iterative_refinement_label(roles: PrecisionRoles) -> str:
    """Return the method label for any iterative-refinement triple."""
    triple = "–".join(
        precision_label(name)
        for name in (roles.factor, roles.work, roles.residual)
    )
    return f"Iterative refinement: {triple}"


def direct_lu_label(roles: PrecisionRoles) -> str:
    """Return the direct-solve label derived from CSV metadata."""
    return f"Direct {precision_label(roles.work)} LU"


def precision_label(precision_name: str) -> str:
    """Return a compact display label for a precision name."""
    if precision_name.startswith("fp"):
        return precision_name.upper()
    if precision_name == "bfloat16":
        return "bfloat16"
    return precision_name


def iterative_refinement_label(roles: PrecisionRoles) -> str:
    """Return the method label for any iterative-refinement triple."""
    triple = "–".join(
        precision_label(name)
        for name in (roles.factor, roles.work, roles.residual)
    )
    return f"Iterative refinement: {triple}"


def direct_lu_label(roles: PrecisionRoles) -> str:
    """Return the direct-solve label derived from CSV metadata."""
    return f"Direct {precision_label(roles.work)} LU"


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
    """Return compact status counts for a method legend label."""
    counts = dataframe["status"].value_counts()
    parts: list[str] = []
    for status, count in counts.items():
        parts.append(f"{int(count)} {STATUS_STYLES[status].code}")
    return ", ".join(parts)


def plot_method_curve(
    axis: Axes,
    dataframe: pd.DataFrame,
    variant: str,
    y_column: str,
) -> None:
    """Plot one method, with status-coded mixed-IR markers."""
    style = METHOD_STYLES[variant]
    axis.plot(
        dataframe["requested_kappa"],
        dataframe[y_column],
        color=style["color"],
        linewidth=1.35,
        zorder=2,
    )

    if variant == "direct-lu":
        available = dataframe[dataframe[y_column].notna()]
        axis.scatter(
            available["requested_kappa"],
            available[y_column],
            marker="s",
            s=24,
            facecolors="white",
            edgecolors=style["color"],
            linewidths=0.85,
            zorder=3,
        )
        return

    for status, status_style in STATUS_STYLES.items():
        if status not in MIXED_IR_STATUS_NAMES:
            continue
        subset = dataframe[
            (dataframe["status"] == status) & dataframe[y_column].notna()
        ]
        if subset.empty:
            continue
        filled = status != "converged"
        axis.scatter(
            subset["requested_kappa"],
            subset[y_column],
            marker=status_style.marker,
            s=32 if status == "converged" else 44,
            facecolors=style["color"] if filled else "white",
            edgecolors=style["color"],
            linewidths=0.9,
            zorder=4,
        )


def make_legend_handles(
    sweeps: dict[str, pd.DataFrame],
    roles: dict[str, PrecisionRoles],
) -> tuple[list[Line2D], list[str]]:
    """Build method and mixed-status legend entries."""
    mixed = sweeps["mixed-ir"]
    direct = sweeps["direct-lu"]
    handles = [
        Line2D([], [], color=METHOD_STYLES["mixed-ir"]["color"], linewidth=1.5),
        Line2D(
            [],
            [],
            color=METHOD_STYLES["direct-lu"]["color"],
            marker="s",
            markerfacecolor="white",
            markersize=5,
            linewidth=1.5,
        ),
    ]
    labels = [
        f"{iterative_refinement_label(roles['mixed-ir'])} "
        f"({status_counts(mixed)})",
        f"{direct_lu_label(roles['direct-lu'])} "
        f"({status_counts(direct)})",
    ]

    present_statuses = set(mixed["status"])
    for status, style in STATUS_STYLES.items():
        if status not in MIXED_IR_STATUS_NAMES:
            continue
        if status not in present_statuses:
            continue
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=style.marker,
                markerfacecolor=(
                    "white"
                    if status == "converged"
                    else METHOD_STYLES["mixed-ir"]["color"]
                ),
                markeredgecolor=METHOD_STYLES["mixed-ir"]["color"],
                markersize=6.5,
            )
        )
        labels.append(f"[{style.code}] {style.description}")

    return handles, labels


def make_figure(
    sweeps: dict[str, pd.DataFrame],
    csv_path: Path,
) -> Figure:
    """Create the two-panel Group D comparison figure."""
    mixed = sweeps["mixed-ir"]
    roles = validate_precision_roles(sweeps, csv_path)
    iterative_roles = roles["mixed-ir"]
    direct_roles = roles["direct-lu"]
    family = str(invariant_value(mixed, "matrix_family", csv_path))
    dimension = int(invariant_value(mixed, "dimension", csv_path))

    factor_boundary = factorization_boundary(iterative_roles.factor)
    iterative_work_roundoff = unit_roundoff(iterative_roles.work)
    direct_roundoff = unit_roundoff(direct_roles.work)
    kappas = mixed["requested_kappa"].to_numpy(dtype=float)
    boundary_is_visible = kappas.min() <= factor_boundary <= kappas.max()
    shared_work_roundoff = math.isclose(
        iterative_work_roundoff,
        direct_roundoff,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    figure, (forward_axis, backward_axis) = plt.subplots(
        2,
        1,
        figsize=(10.8, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1.12, 1.0)},
    )

    for axis in (forward_axis, backward_axis):
        configure_loglog_axis(axis)
        if boundary_is_visible:
            axis.axvline(
                factor_boundary,
                color="#333333",
                linestyle="--",
                linewidth=1.2,
                zorder=1,
            )

        if shared_work_roundoff:
            axis.axhline(
                iterative_work_roundoff,
                color="#666666",
                linestyle=":",
                linewidth=1.1,
                zorder=1,
            )
        else:
            axis.axhline(
                iterative_work_roundoff,
                color=METHOD_STYLES["mixed-ir"]["color"],
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
                zorder=1,
            )
            axis.axhline(
                direct_roundoff,
                color=METHOD_STYLES["direct-lu"]["color"],
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
                zorder=1,
            )

    for variant in EXPECTED_VARIANTS:
        plot_method_curve(
            forward_axis,
            sweeps[variant],
            variant,
            "final_forward_error_inf",
        )
        plot_method_curve(
            backward_axis,
            sweeps[variant],
            variant,
            "final_backward_error_inf",
        )

    # A backward-stable direct solve typically has forward error O(kappa*u).
    # This is a slope reference, not a pointwise error bound.
    forward_axis.plot(
        kappas,
        kappas * direct_roundoff,
        color="#777777",
        linestyle="-.",
        linewidth=1.05,
        zorder=1,
    )

    forward_axis.set_ylabel("Relative forward error")
    forward_axis.set_title(
        "(a) Final relative forward error, infinity norm", loc="left"
    )

    backward_axis.set_ylabel("Relative backward error")
    backward_axis.set_xlabel(r"Requested condition number $\kappa$")
    backward_axis.set_title(
        "(b) Final normwise backward error, infinity norm", loc="left"
    )

    figure.suptitle(
        "Direct-solve comparison: iterative refinement versus direct LU",
        fontsize=15,
        y=0.985,
    )
    subtitle = (
        f"IR = "
        f"{precision_label(iterative_roles.factor)}–"
        f"{precision_label(iterative_roles.work)}–"
        f"{precision_label(iterative_roles.residual)}; "
        f"direct = {precision_label(direct_roles.work)}; "
        f"{family.replace('-', ' ')}, n = {dimension}, "
        f"measurement = {precision_label(iterative_roles.measure)}"
    )
    figure.text(0.5, 0.949, subtitle, ha="center", va="top", fontsize=10)

    boundary_text = format_number(factor_boundary)
    if boundary_is_visible:
        boundary_reference = (
            rf"Dashed vertical: $\kappa_*=1/u_f={boundary_text}$"
        )
    else:
        boundary_reference = (
            rf"IR factorization boundary: "
            rf"$\kappa_*=1/u_f={boundary_text}$ (outside sweep)"
        )

    if shared_work_roundoff:
        roundoff_reference = (
            rf"Dotted horizontal: $u={format_number(direct_roundoff)}$"
        )
    else:
        roundoff_reference = (
            rf"Dotted: $u_{{\mathrm{{IR}}}}="
            rf"{format_number(iterative_work_roundoff)}$, "
            rf"$u_{{\mathrm{{direct}}}}={format_number(direct_roundoff)}$"
        )

    reference_text = (
        boundary_reference
        + "    "
        + roundoff_reference
        + rf"    Dash-dot (forward): $\kappa u_{{\mathrm{{direct}}}}$"
    )
    figure.text(0.5, 0.919, reference_text, ha="center", va="top", fontsize=9)

    handles, labels = make_legend_handles(sweeps, roles)
    figure.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=2,
        frameon=False,
        fontsize=9,
        handletextpad=0.55,
        columnspacing=1.4,
    )

    missing_errors = sum(
        int(sweep[["final_forward_error_inf", "final_backward_error_inf"]]
            .isna()
            .any(axis=1)
            .sum())
        for sweep in sweeps.values()
    )
    if missing_errors:
        forward_axis.text(
            0.015,
            0.965,
            f"{missing_errors} run(s) have no returned error measurement",
            transform=forward_axis.transAxes,
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
        top=0.80,
        bottom=0.10,
        left=0.12,
        right=0.98,
        hspace=0.28,
    )
    return figure


def main() -> int:
    """Plot every selected Group D comparison dataset."""
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
        sweeps = read_comparison(csv_path)
        figure = make_figure(sweeps, csv_path)
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