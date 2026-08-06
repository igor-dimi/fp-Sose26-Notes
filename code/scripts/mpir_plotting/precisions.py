"""Floating-point precision metadata shared by experiment plots.

The experiment framework uses ``u = 2**(-p)``, where ``p`` is the recorded
number of significand bits.  Consequently, the usual factorization boundary
is ``kappa_* = 1 / u_f = 2**p``.

This module contains numerical facts and compact display labels only.  Choices
such as colors, line styles, and whether a reference line belongs in a given
figure remain the responsibility of the individual plotting scripts.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PrecisionInfo:
    """Metadata and derived quantities for one supported precision."""

    name: str
    label: str
    significand_bits: int

    @property
    def unit_roundoff(self) -> float:
        """Return the unit roundoff ``u = 2**(-p)``."""
        return math.ldexp(1.0, -self.significand_bits)

    @property
    def condition_boundary(self) -> float:
        """Return the boundary ``kappa_* = 1/u = 2**p``."""
        return math.ldexp(1.0, self.significand_bits)


PRECISIONS: dict[str, PrecisionInfo] = {
    "fp8": PrecisionInfo("fp8", "FP8", 4),
    "bfloat16": PrecisionInfo("bfloat16", "bfloat16", 8),
    "fp16": PrecisionInfo("fp16", "FP16", 11),
    "fp32": PrecisionInfo("fp32", "FP32", 24),
    "fp64": PrecisionInfo("fp64", "FP64", 53),
    "fp128": PrecisionInfo("fp128", "FP128", 128),
    "fp256": PrecisionInfo("fp256", "FP256", 256),
}


def precision_info(name: str) -> PrecisionInfo:
    """Return metadata for a supported precision name.

    Names are matched case-insensitively and may contain surrounding
    whitespace.  A clear ``ValueError`` is raised for an unknown name.
    """
    normalized_name = name.strip().lower()

    try:
        return PRECISIONS[normalized_name]
    except KeyError as error:
        supported = ", ".join(PRECISIONS)
        raise ValueError(
            f"Unknown precision {name!r}; expected one of: {supported}."
        ) from error


def is_supported_precision(name: str) -> bool:
    """Return whether ``name`` identifies a supported precision."""
    return name.strip().lower() in PRECISIONS


def precision_label(name: str) -> str:
    """Return the compact display label for a supported precision."""
    return precision_info(name).label


def unit_roundoff(name: str) -> float:
    """Return ``u = 2**(-p)`` for a supported precision."""
    return precision_info(name).unit_roundoff


def factorization_boundary(name: str) -> float:
    """Return ``kappa_* = 1/u = 2**p`` for a supported precision."""
    return precision_info(name).condition_boundary


__all__ = [
    "PRECISIONS",
    "PrecisionInfo",
    "factorization_boundary",
    "is_supported_precision",
    "precision_info",
    "precision_label",
    "unit_roundoff",
]
