"""Learned parameter types."""

from __future__ import annotations

from typing import Optional

from .constants import ChangeModel
from .parameter import CompoundParameter, Parameter


class LearnedBias(CompoundParameter):
    def __init__(
        self,
        target: Parameter,
        expected_change: Optional[int] = None,
        change_model: Optional[ChangeModel] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.target = target
        self.expected_change = expected_change
        self.change_model = change_model
        self.bias = target.clone(name="bias")
        self.applied = target.clone(name="applied")
        self.add_child("bias", self.bias)
        self.add_child("applied", self.applied)
