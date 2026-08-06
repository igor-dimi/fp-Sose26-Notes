"""Shared CSV-loading and validation primitives for experiment plots.

This module contains only validation operations that are independent of a
particular experiment.  Rules such as permitted variants, valid termination
statuses, or required iteration sequences belong in the plotting script that
understands the dataset's semantics.
"""

from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: Collection[str],
    path: Path,
) -> None:
    """Require every named column to be present in ``dataframe``."""
    missing_columns = set(required_columns) - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_csv_checked(
    path: Path,
    required_columns: Collection[str],
    *,
    allow_empty: bool = False,
    **read_csv_options: Any,
) -> pd.DataFrame:
    """Read a CSV file and validate its basic tabular structure.

    Additional keyword arguments are forwarded to :func:`pandas.read_csv`.
    By default, a header-only CSV is rejected because it contains no dataset
    to plot.  Pass ``allow_empty=True`` only when an experiment explicitly
    gives an empty table a meaning.
    """
    dataframe = pd.read_csv(path, **read_csv_options)
    require_columns(dataframe, required_columns, path)

    if dataframe.empty and not allow_empty:
        raise ValueError(f"{path} contains no data rows.")

    return dataframe


def coerce_numeric_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    path: Path,
    *,
    require_complete: bool = False,
) -> None:
    """Convert selected columns to numeric values in place.

    Values that cannot be converted become ``NaN``.  This preserves the
    distinction needed by several experiments, where an unavailable error or
    a status-only row is valid and is handled later by experiment-specific
    validation.

    If ``require_complete`` is true, reject every selected column containing
    a missing or unconvertible value after conversion.
    """
    numeric_columns = tuple(columns)
    require_columns(dataframe, numeric_columns, path)

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    if not require_complete:
        return

    invalid_columns = [
        column
        for column in numeric_columns
        if dataframe[column].isna().any()
    ]
    if invalid_columns:
        invalid = ", ".join(invalid_columns)
        raise ValueError(
            f"{path} contains missing or invalid numeric data in: {invalid}"
        )


def invariant_value(
    dataframe: pd.DataFrame,
    column: str,
    path: Path,
) -> Any:
    """Return the single non-missing value in an invariant column.

    Missing entries are ignored, matching the behavior of the plotting
    scripts' former ``one_value`` helpers.  The function still rejects a
    column containing no value or more than one distinct non-missing value.
    """
    require_columns(dataframe, (column,), path)

    values = dataframe[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"{path} must contain exactly one value for {column}; "
            f"found {len(values)}."
        )

    return values[0]


def require_invariant_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    path: Path,
) -> None:
    """Require each named column to have one invariant non-missing value."""
    for column in columns:
        invariant_value(dataframe, column, path)
