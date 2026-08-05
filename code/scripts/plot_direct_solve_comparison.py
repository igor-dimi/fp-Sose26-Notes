#!/usr/bin/env python3
"""Plot the Group D direct-solve comparison.

By default, the script reads every direct-solve-comparison CSV from
``results/raw/direct_comparison`` and writes one two-panel log-log
figure per dataset to ``results/plots/direct_comparison``.

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


CSV_PATTERN = "direct-solve-comparison__*.csv"
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

# Number of significand bits p. The experiment uses u = 2**(-p), so the
# low-precision factorization boundary is kappa_* = 1/u_f = 2**p.
PRECISION_SIGNIFICAND_BITS = {
    "fp8": 4,
    "bfloat16": 8,
    "fp16": 11,
    "fp32": 24,
    "fp64": 53,
    "fp128": 128,
    "fp256": 256,
}

METHOD_STYLES = {
    "mixed-ir": {
        "label": "Mixed IR: FP32–FP64–FP128",
        "color": "#0072B2",
    },
    "direct-lu": {
        "label": "Direct FP64 LU",
        "color": "#D55E00",
    },
}


@dataclass(frozen=True)
class StatusStyle:
    """Visual and textual encoding for one mixed-IR status."""

    code: str
    description: str
    marker: str


MIXED_STATUS_STYLES = {
    "converged": StatusStyle("C", "converged", "o"),
    "max-iterations": StatusStyle("M", "maximum iterations", "^"),
    "diverged": StatusStyle("D", "diverged", "X"),
    "stagnated": StatusStyle("S", "stagnated", "P"),
    "non-finite": StatusStyle("N", "non-finite value", "s"),
    "factorization-input-non-finite": StatusStyle(
        "F", "non-finite factorization input", "D"
    ),
}


def parse_arguments() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description=(
            "Compare Group D mixed-precision iterative refinement with a "
            "direct FP64 LU solve in forward- and backward-error panels."
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
        raw_root / "direct_comparison" / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find input {path}. Checked:\n{checked}")


def discover_csv_files(inputs: Iterable[Path], raw_root: Path) -> list[Path]:
    """Discover Group D comparison CSV files."""
    input_paths = list(inputs)
    if not input_paths:
        input_directory = raw_root / "direct_comparison"
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


def validate_precision_roles(
    sweeps: dict[str, pd.DataFrame],
    csv_path: Path,
) -> None:
    """Check the precision configurations prescribed for Group D."""
    expected = {
        "mixed-ir": ("fp32", "fp64", "fp128"),
        "direct-lu": ("fp64", "fp64", "none"),
    }
    for variant, expected_roles in expected.items():
        subset = sweeps[variant]
        actual_roles = tuple(
            str(one_metadata_value(subset, column, csv_path))
            for column in (
                "factor_precision",
                "work_precision",
                "residual_precision",
            )
        )
        if actual_roles != expected_roles:
            raise ValueError(
                f"{csv_path}: {variant} has precision roles {actual_roles}; "
                f"expected {expected_roles}."
            )


def read_comparison(csv_path: Path) -> dict[str, pd.DataFrame]:
    """Read and validate one Group D comparison CSV."""
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

    dataframe["variant"] = dataframe["variant"].astype(str).str.strip()
    dataframe["status"] = dataframe["status"].astype(str).str.strip()
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
            one_metadata_value(dataframe, column, csv_path)

    if str(one_metadata_value(dataframe, "experiment", csv_path)) != (
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
        set(sweeps["mixed-ir"]["status"]) - set(MIXED_STATUS_STYLES)
    )
    if unknown_mixed_statuses:
        raise ValueError(
            f"{csv_path} contains unknown mixed-IR status value(s): "
            f"{', '.join(map(repr, unknown_mixed_statuses))}."
        )

    validate_precision_roles(sweeps, csv_path)
    return sweeps


def unit_roundoff(precision_name: str) -> float:
    """Return unit roundoff for a supported precision."""
    try:
        significand_bits = PRECISION_SIGNIFICAND_BITS[precision_name]
    except KeyError as error:
        supported = ", ".join(PRECISION_SIGNIFICAND_BITS)
        raise ValueError(
            f"Unknown precision {precision_name!r}; expected one of: "
            f"{supported}."
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


def status_counts(dataframe: pd.DataFrame, mixed: bool) -> str:
    """Return compact status counts for a method legend label."""
    counts = dataframe["status"].value_counts()
    parts: list[str] = []
    for status, count in counts.items():
        if mixed and status in MIXED_STATUS_STYLES:
            name = MIXED_STATUS_STYLES[status].code
        else:
            name = str(status)
        parts.append(f"{int(count)} {name}")
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

    for status, status_style in MIXED_STATUS_STYLES.items():
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
        f"{METHOD_STYLES['mixed-ir']['label']} "
        f"({status_counts(mixed, mixed=True)})",
        f"{METHOD_STYLES['direct-lu']['label']} "
        f"({status_counts(direct, mixed=False)})",
    ]

    present_statuses = set(mixed["status"])
    for status, style in MIXED_STATUS_STYLES.items():
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
    factor = str(one_metadata_value(mixed, "factor_precision", csv_path))
    work = str(one_metadata_value(mixed, "work_precision", csv_path))
    measure = str(one_metadata_value(mixed, "measure_precision", csv_path))
    family = str(one_metadata_value(mixed, "matrix_family", csv_path))
    dimension = int(one_metadata_value(mixed, "dimension", csv_path))

    factor_boundary = 1.0 / unit_roundoff(factor)
    work_roundoff = unit_roundoff(work)
    kappas = mixed["requested_kappa"].to_numpy(dtype=float)

    figure, (forward_axis, backward_axis) = plt.subplots(
        2,
        1,
        figsize=(10.8, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1.12, 1.0)},
    )

    for axis in (forward_axis, backward_axis):
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
        kappas * work_roundoff,
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
        "Direct-solve comparison: mixed IR versus direct FP64 LU",
        fontsize=15,
        y=0.985,
    )
    subtitle = (
        f"Mixed IR = {factor.upper()}–{work.upper()}–FP128; "
        f"{family.replace('-', ' ')}, n = {dimension}, "
        f"measurement = {measure.upper()}"
    )
    figure.text(0.5, 0.949, subtitle, ha="center", va="top", fontsize=10)

    boundary_text = format_number(factor_boundary)
    roundoff_text = format_number(work_roundoff)
    reference_text = (
        rf"Dashed vertical: $\kappa_*=1/u_f={boundary_text}$"
        rf"    Dotted horizontal: $u_{{\mathrm{{work}}}}={roundoff_text}$"
        rf"    Dash-dot (forward): reference slope $\kappa u_{{\mathrm{{work}}}}$"
    )
    figure.text(0.5, 0.919, reference_text, ha="center", va="top", fontsize=9)

    handles, labels = make_legend_handles(sweeps)
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
        relative_parent = Path("direct_comparison")

    output_directory = plots_root / relative_parent
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / f"{csv_path.stem}.{output_format}"


def main() -> int:
    """Plot every selected Group D comparison dataset."""
    args = parse_arguments()
    raw_root, plots_root = resolve_roots(args)
    csv_files = discover_csv_files(args.inputs, raw_root)

    for csv_path in csv_files:
        sweeps = read_comparison(csv_path)
        figure = make_figure(sweeps, csv_path)
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
