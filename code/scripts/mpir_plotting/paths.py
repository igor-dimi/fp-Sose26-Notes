"""Shared path handling for mixed-precision experiment plots.

The package is expected to live under ``code/scripts/mpir_plotting`` while
experiment data lives under the repository-level ``results`` directory.
"""

from collections.abc import Iterable
from pathlib import Path


def infer_repository_root() -> Path:
    """Return the repository root for this package's conventional location.

    For ``<repository>/code/scripts/mpir_plotting/paths.py``, the repository
    root is the fourth parent of this file.
    """
    return Path(__file__).resolve().parents[3]


def resolve_results_roots(
    raw_root: Path | None = None,
    plots_root: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve the raw-data and plot-output roots.

    Explicit paths are resolved relative to the current working directory.
    Missing paths default to ``<repository>/results/raw`` and
    ``<repository>/results/plots``.
    """
    repository_root = infer_repository_root()

    resolved_raw_root = (
        raw_root.expanduser().resolve()
        if raw_root is not None
        else repository_root / "results" / "raw"
    )
    resolved_plots_root = (
        plots_root.expanduser().resolve()
        if plots_root is not None
        else repository_root / "results" / "plots"
    )

    return resolved_raw_root, resolved_plots_root


def resolve_input_path(
    path: Path,
    raw_root: Path,
    default_subdirectory: Path | str,
) -> Path:
    """Resolve an explicit input against the usual experiment locations.

    Inputs are checked in this order:

    1. as supplied, relative to the current working directory if necessary;
    2. relative to the raw-results root;
    3. relative to the experiment's default raw-results subdirectory.
    """
    default_subdirectory = Path(default_subdirectory)
    candidates = (
        path.expanduser(),
        raw_root / path,
        raw_root / default_subdirectory / path,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find input {path}. Checked:\n{checked}")


def discover_csv_files(
    inputs: Iterable[Path],
    raw_root: Path,
    default_subdirectory: Path | str,
    pattern: str,
) -> list[Path]:
    """Discover CSV inputs for one experiment.

    With no explicit inputs, files matching ``pattern`` are read from the
    experiment's default subdirectory. Explicit directory inputs are searched
    recursively; explicit CSV files are accepted directly.
    """
    default_subdirectory = Path(default_subdirectory)
    input_paths = list(inputs)

    if not input_paths:
        input_directory = raw_root / default_subdirectory
        if not input_directory.is_dir():
            raise FileNotFoundError(
                f"Default input directory does not exist: {input_directory}"
            )
        csv_files = list(input_directory.glob(pattern))
    else:
        csv_files: list[Path] = []
        for input_path in input_paths:
            resolved_path = resolve_input_path(
                input_path,
                raw_root,
                default_subdirectory,
            )

            if resolved_path.is_dir():
                csv_files.extend(resolved_path.rglob(pattern))
            elif resolved_path.is_file() and resolved_path.suffix.lower() == ".csv":
                csv_files.append(resolved_path)
            else:
                raise ValueError(f"Input file is not a CSV file: {resolved_path}")

    unique_files = sorted({path.resolve() for path in csv_files})
    if not unique_files:
        raise FileNotFoundError(f"No files matching {pattern} were found.")

    return unique_files


def mirrored_plot_directory(
    csv_path: Path,
    raw_root: Path,
    plots_root: Path,
    fallback_subdirectory: Path | str,
) -> Path:
    """Map a CSV's raw-results directory into the plot-results tree.

    If ``csv_path`` is outside ``raw_root``, use ``fallback_subdirectory``.
    The returned directory is not created by this function.
    """
    try:
        relative_directory = csv_path.resolve().parent.relative_to(
            raw_root.resolve()
        )
    except ValueError:
        relative_directory = Path(fallback_subdirectory)

    return plots_root / relative_directory


def mirrored_plot_path(
    csv_path: Path,
    raw_root: Path,
    plots_root: Path,
    output_format: str,
    fallback_subdirectory: Path | str,
    *,
    suffix: str = "",
) -> Path:
    """Construct and prepare a mirrored output path for one CSV dataset.

    ``suffix`` is inserted after the CSV stem. For example, passing
    ``suffix="__diagnostics"`` produces ``<stem>__diagnostics.png``.
    """
    normalized_format = output_format.removeprefix(".")
    if not normalized_format:
        raise ValueError("output_format must not be empty")

    output_directory = mirrored_plot_directory(
        csv_path,
        raw_root,
        plots_root,
        fallback_subdirectory,
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory / f"{csv_path.stem}{suffix}.{normalized_format}"
