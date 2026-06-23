"""Design generators. Phase 1: factorial designs."""

from __future__ import annotations

from .factorial import fractional_factorial, full_factorial, plackett_burman

__all__ = ["fractional_factorial", "full_factorial", "plackett_burman"]
