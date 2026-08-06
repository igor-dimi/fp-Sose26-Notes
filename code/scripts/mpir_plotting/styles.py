"""Shared visual encodings for experiment termination statuses.

The status name written to a CSV is a semantic value.  Its compact code,
description, marker, and color should therefore be consistent across every
plotting script.  Styles for experiment variants or numerical methods remain
local to the script that defines those concepts.

The registry includes both the current mixed-IR statuses and the direct-solve
statuses used by the comparison experiment.  The detailed mixed-IR failure
codes are deliberately distinct, so a legend never assigns one code to two
different termination reasons.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class StatusStyle:
    """Textual and visual encoding for one termination status."""

    code: str
    description: str
    marker: str
    color: str


# Colors follow the colorblind-safe palette already used by the experiment
# plots.  Detailed numerical failures share black and remain distinguishable
# through their marker and compact code.
STATUS_STYLES: dict[str, StatusStyle] = {
    "converged": StatusStyle("C", "converged", "o", "#0072B2"),
    "max-iterations": StatusStyle(
        "M", "maximum iterations", "^", "#E69F00"
    ),
    "diverged": StatusStyle("D", "diverged", "X", "#D55E00"),
    "stagnated": StatusStyle("S", "stagnated", "P", "#CC79A7"),
    "non-finite": StatusStyle("N", "non-finite value", "s", "#000000"),
    "factorization-input-non-finite": StatusStyle(
        "FI", "non-finite factorization input", "D", "#000000"
    ),
    "factorization-failure": StatusStyle(
        "FF", "factorization failure", "v", "#000000"
    ),
    "initial-solution-non-finite": StatusStyle(
        "XN", "non-finite initial solution", "<", "#000000"
    ),
    "residual-non-finite": StatusStyle(
        "RN", "non-finite residual", ">", "#000000"
    ),
    "correction-non-finite": StatusStyle(
        "CN", "non-finite correction", "d", "#000000"
    ),
    "iterate-non-finite": StatusStyle(
        "IN", "non-finite iterate", "*", "#000000"
    ),
    "residual-conversion-underflow": StatusStyle(
        "U", "residual-conversion underflow", "h", "#009E73"
    ),
    # Direct-LU statuses used by exp_direct_solve_comparison.cc.
    "success": StatusStyle("OK", "successful direct solve", "o", "#0072B2"),
    "failure": StatusStyle("E", "direct-solve failure", "X", "#000000"),
    "input-non-finite": StatusStyle(
        "DI", "non-finite direct-solve input", "s", "#000000"
    ),
    "factorization-non-finite": StatusStyle(
        "DF", "non-finite direct factorization", "D", "#000000"
    ),
    "solution-non-finite": StatusStyle(
        "DS", "non-finite direct solution", "v", "#000000"
    ),
}


MIXED_IR_STATUS_NAMES = frozenset(
    {
        "converged",
        "max-iterations",
        "diverged",
        "stagnated",
        "non-finite",
        "factorization-input-non-finite",
        "factorization-failure",
        "initial-solution-non-finite",
        "residual-non-finite",
        "correction-non-finite",
        "iterate-non-finite",
        "residual-conversion-underflow",
    }
)

DIRECT_SOLVE_STATUS_NAMES = frozenset(
    {
        "success",
        "failure",
        "input-non-finite",
        "factorization-non-finite",
        "solution-non-finite",
    }
)


def _normalize_status(status: str) -> str:
    """Normalize a status name for registry lookup."""
    return status.strip().lower()


def status_style(status: str) -> StatusStyle:
    """Return the shared style for a known status name.

    Status names are matched case-insensitively and may contain surrounding
    whitespace.  Unknown names raise a clear ``ValueError``.
    """
    normalized_status = _normalize_status(status)

    try:
        return STATUS_STYLES[normalized_status]
    except KeyError as error:
        supported = ", ".join(STATUS_STYLES)
        raise ValueError(
            f"Unknown termination status {status!r}; expected one of: "
            f"{supported}."
        ) from error


def is_supported_status(status: str) -> bool:
    """Return whether ``status`` has a shared style."""
    return _normalize_status(status) in STATUS_STYLES


def status_code(status: str) -> str:
    """Return the compact legend code for ``status``."""
    return status_style(status).code


def status_description(status: str) -> str:
    """Return the human-readable description for ``status``."""
    return status_style(status).description


def status_key(statuses: Iterable[str]) -> str:
    """Describe the distinct statuses present in a dataset.

    Definitions follow registry order rather than input order, which keeps
    legends stable when CSV row order changes.
    """
    normalized_statuses = {_normalize_status(status) for status in statuses}

    unknown_statuses = normalized_statuses - set(STATUS_STYLES)
    if unknown_statuses:
        unknown = ", ".join(repr(status) for status in sorted(unknown_statuses))
        raise ValueError(f"Unknown termination status value(s): {unknown}.")

    definitions = [
        f"{style.code} = {style.description}"
        for name, style in STATUS_STYLES.items()
        if name in normalized_statuses
    ]

    if not definitions:
        return "Status: none"

    return "Status: " + "; ".join(definitions)


__all__ = [
    "DIRECT_SOLVE_STATUS_NAMES",
    "MIXED_IR_STATUS_NAMES",
    "STATUS_STYLES",
    "StatusStyle",
    "is_supported_status",
    "status_code",
    "status_description",
    "status_key",
    "status_style",
]