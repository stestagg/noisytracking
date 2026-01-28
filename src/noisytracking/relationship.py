"""Relationship definitions for parameter connections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from typing import TYPE_CHECKING

from .constants import OutlierHandling, Rel

if TYPE_CHECKING:
    from .parameter import Parameter


@dataclass
class Relationship:
    source: Parameter
    rel: Optional[Rel]
    outlier_handling: Optional[OutlierHandling]
