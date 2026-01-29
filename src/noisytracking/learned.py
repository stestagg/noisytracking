"""Learned parameter types."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Tuple, List

from .constants import ChangeModel, Rel
from .parameter import CompoundParameter, Parameter

if TYPE_CHECKING:
    from .op_graph import OpGraphHelper, OpNode
    from .parameter import ScalarParameter


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

    def get_custom_ops(
        self,
        helper: "OpGraphHelper",
        root_name: str,
        leaf_path: Tuple[str, ...],
        leaf: Optional["ScalarParameter"],
    ) -> List["OpNode"]:
        if not leaf_path or leaf_path[0] != "applied" or leaf is None:
            return []

        applied_leaf_path = leaf_path[1:]
        target_inputs = helper.build_inputs_from_relationships(
            self.applied,
            applied_leaf_path,
            leaf,
        )

        if not target_inputs:
            target_node = helper.build_source_node(
                source_param=self.target,
                source_subpath=applied_leaf_path,
                target_units=leaf.units,
                rel_type=Rel.absolute,
                outlier=None,
                include_update=True,
            )
            if target_node is not None:
                target_inputs = [target_node]

        if not target_inputs:
            return []

        bias_leaf = helper.get_leaf(self.bias, applied_leaf_path)
        bias_node = helper.build_value_node(
            root_name,
            self,
            ("bias",) + applied_leaf_path,
            bias_leaf,
            is_output=False,
        )

        bias_nodes: List["OpNode"] = []
        for target_node in target_inputs:
            bias_nodes.append(helper.create_bias_node([target_node, bias_node]))

        return bias_nodes
