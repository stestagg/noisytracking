"""Operation graph builder for BuildModel."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from .constants import OutlierHandling, Rel
from .units import ParsedUnit, Exponent

if TYPE_CHECKING:
    from .builder import BuildModel
    from .parameter import Parameter, ScalarParameter


# ============================================================================
# Operation Node Classes
# ============================================================================


@dataclass
class OpNode(ABC):
    """Base class for all operation graph nodes."""

    id: str
    inputs: List["OpNode"] = field(default_factory=list)


# Input Nodes


@dataclass
class SensorInput(OpNode):
    """Input node representing a sensor measurement."""

    sensor_name: str = ""
    parameter_path: Tuple[str, ...] = ()
    units: Optional[ParsedUnit] = None


@dataclass
class LearnedValue(OpNode):
    """Value node representing a learned parameter."""

    learned_name: str = ""
    parameter_path: Tuple[str, ...] = ()
    units: Optional[ParsedUnit] = None


# Operation Nodes


@dataclass
class Scale(OpNode):
    """Scales input by a factor (for unit conversion)."""

    scale_factor: float = 1.0
    source_units: Optional[ParsedUnit] = None
    target_units: Optional[ParsedUnit] = None


@dataclass
class Bias(OpNode):
    """Adds a bias value to the base input."""


@dataclass
class State(OpNode):
    """Holds accumulated state for delta/per_second relationships."""

    state_name: str = ""
    initial_value: float = 0.0
    units: Optional[ParsedUnit] = None


@dataclass
class StateUpdate(OpNode):
    """Updates state: new_value = state + scaled_delta."""

    state_ref: Optional[State] = None
    update_type: Optional[Rel] = None


@dataclass
class Update(OpNode):
    """Measurement update / fusion."""

    outlier_handling: Optional[OutlierHandling] = None


# Output Nodes


@dataclass
class PredictionOutput(OpNode):
    """Output node for a predicted scalar parameter."""

    parameter_name: str = ""
    path: Tuple[str, ...] = ()


class OpGraph(Mapping[str, PredictionOutput]):
    """Operation graph for a BuildModel."""

    def __init__(self, outputs: Dict[str, PredictionOutput]) -> None:
        self._outputs = dict(outputs)

    def __getitem__(self, key: str) -> PredictionOutput:
        return self._outputs[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self._outputs)

    def __len__(self) -> int:
        return len(self._outputs)

    def items(self) -> Iterable[Tuple[str, PredictionOutput]]:
        return self._outputs.items()

    def values(self) -> Iterable[PredictionOutput]:
        return self._outputs.values()

    @property
    def outputs(self) -> Dict[str, PredictionOutput]:
        return dict(self._outputs)

    @classmethod
    def from_model(cls, model: "BuildModel") -> "OpGraph":
        """Build an operation graph from a BuildModel."""

        builder = OpGraphBuilder(model)
        return cls(builder.build())

    def as_mermaid(self) -> str:
        """Render the operation graph as a Mermaid flowchart."""

        return render_op_graph_mermaid(self._outputs)


# ============================================================================
# Helper Functions
# ============================================================================


def collect_scalar_leaves(
    param: "Parameter", path: Tuple[str, ...] = ()
) -> List[Tuple[Tuple[str, ...], "ScalarParameter"]]:
    """Recursively find all ScalarParameter leaves with their paths.

    Args:
        param: The parameter to search.
        path: The current path from the root parameter.

    Returns:
        List of (path, ScalarParameter) tuples.
    """

    from .parameter import ScalarParameter

    results: List[Tuple[Tuple[str, ...], ScalarParameter]] = []

    if isinstance(param, ScalarParameter):
        results.append((path, param))
    else:
        for child_name, child in param.child_parameters.items():
            child_path = path + (child_name,)
            results.extend(collect_scalar_leaves(child, child_path))

    return results


def find_source_info(
    source: "Parameter", model: "BuildModel"
) -> Tuple[str, str, Tuple[str, ...]]:
    """Determine if source is a sensor or learned parameter and get its path.

    Args:
        source: The source parameter.
        model: The BuildModel containing sensors and learned parameters.

    Returns:
        Tuple of (source_type, name, path_within_source)
        source_type is either "sensor" or "learned".
    """

    # Check sensors
    for sensor_name, sensor_def in model._sensors.items():
        path = get_path_from_ancestor(sensor_def.parameter, source)
        if path is not None:
            return ("sensor", sensor_name, path)

    # Check learned parameters
    for learned_name, learned_param in model._learned.items():
        path = get_path_from_ancestor(learned_param, source)
        if path is not None:
            return ("learned", learned_name, path)

    raise ValueError(f"Source parameter {source.name} not found in sensors or learned")


def is_descendant_of(child: "Parameter", ancestor: "Parameter") -> bool:
    """Check if child is a descendant of ancestor (or is ancestor itself)."""

    if child is ancestor:
        return True

    for _, grandchild in ancestor.child_parameters.items():
        if is_descendant_of(child, grandchild):
            return True

    return False


def get_path_from_ancestor(
    ancestor: "Parameter", descendant: "Parameter"
) -> Optional[Tuple[str, ...]]:
    """Get navigation path from ancestor to descendant.

    Args:
        ancestor: The ancestor parameter.
        descendant: The descendant parameter.

    Returns:
        Tuple of child names to navigate from ancestor to descendant,
        or None if descendant is not a descendant of ancestor.
    """

    if ancestor is descendant:
        return ()

    for child_name, child in ancestor.child_parameters.items():
        sub_path = get_path_from_ancestor(child, descendant)
        if sub_path is not None:
            return (child_name,) + sub_path

    return None


def compute_scale_factor(
    source_units: Optional[ParsedUnit],
    target_units: Optional[ParsedUnit],
    rel: Optional[Rel] = None,
) -> float:
    """Calculate unit conversion factor from source to target units.

    Args:
        source_units: The source unit.
        target_units: The target unit.
        rel: The relationship type (affects time dimension handling).

    Returns:
        Scale factor to multiply source values by.
    """

    if source_units is None or target_units is None:
        return 1.0

    if not source_units.terms or not target_units.terms:
        return 1.0

    # Compute conversion: for each dimension, compare factors
    # source * source_factor = base = target * target_factor
    # So: source * (source_factor / target_factor) = target value

    source_factor = 1.0
    target_factor = 1.0

    for term in source_units.terms:
        factor = term.unit.factor
        if term.exponent == Exponent.NEGATIVE:
            source_factor /= factor
        else:
            source_factor *= factor

    for term in target_units.terms:
        factor = term.unit.factor
        if term.exponent == Exponent.NEGATIVE:
            target_factor /= factor
        else:
            target_factor *= factor

    return source_factor / target_factor


def navigate_to_leaf(
    param: "Parameter", path: Tuple[str, ...]
) -> Optional["Parameter"]:
    """Navigate from a parameter to a child at the given path.

    Args:
        param: The starting parameter.
        path: The path to navigate.

    Returns:
        The parameter at the path, or None if not found.
    """

    current = param
    for name in path:
        children = current.child_parameters
        if name not in children:
            return None
        current = children[name]
    return current


def collect_applicable_relationships(
    root: "Parameter",
    leaf_path: Tuple[str, ...],
    leaf: Optional["ScalarParameter"],
) -> List[Tuple["Parameter", "Parameter", Optional[Rel], Optional[OutlierHandling], Tuple[str, ...]]]:
    """Gather relationships affecting a leaf parameter.

    Walks from root through the path to leaf, collecting relationships.
    For relationships on ancestor parameters, we need to find the corresponding
    path within the source parameter.

    Args:
        root: The root predicted parameter.
        leaf_path: Path from root to leaf.
        leaf: The leaf ScalarParameter.

    Returns:
        List of (target, source, rel, outlier_handling, source_subpath) tuples.
        source_subpath is the path from the source to its corresponding leaf.
    """

    results: List[
        Tuple["Parameter", "Parameter", Optional[Rel], Optional[OutlierHandling], Tuple[str, ...]]
    ] = []

    # Walk the path, collecting relationships from each parameter
    current = root
    for i, name in enumerate(leaf_path + (None,)):  # type: ignore
        # Get relationships at this level
        for rel_obj in current._relationships:
            # For relationships at ancestor level, we need the remaining path
            remaining_path = leaf_path[i:] if name is not None else ()
            results.append(
                (
                    current,
                    rel_obj.source,
                    rel_obj.rel,
                    rel_obj.outlier_handling,
                    remaining_path,
                )
            )

        # Navigate to next child
        if name is not None:
            children = current.child_parameters
            if name in children:
                current = children[name]
            else:
                break

    return results


class OpGraphHelper:
    """Helper for custom operation graph construction."""

    def __init__(self, builder: "OpGraphBuilder", root_name: str) -> None:
        self._builder = builder
        self._root_name = root_name

    def build_value_node(
        self,
        root_name: str,
        root_param: "Parameter",
        leaf_path: Tuple[str, ...],
        leaf: Optional["ScalarParameter"],
        *,
        is_output: bool,
    ) -> OpNode:
        return self._builder.build_value_node(
            root_name, root_param, leaf_path, leaf, is_output=is_output
        )

    def build_inputs_from_relationships(
        self,
        root_param: "Parameter",
        leaf_path: Tuple[str, ...],
        leaf: Optional["ScalarParameter"],
    ) -> List[OpNode]:
        return self._builder.build_relationship_inputs(root_param, leaf_path, leaf)

    def build_source_node(
        self,
        source_param: "Parameter",
        source_subpath: Tuple[str, ...],
        target_units: Optional[ParsedUnit],
        rel_type: Optional[Rel],
        outlier: Optional[OutlierHandling],
        *,
        include_update: bool = True,
    ) -> Optional[OpNode]:
        return self._builder.build_source_node(
            source_param,
            source_subpath,
            target_units,
            rel_type,
            outlier,
            include_update=include_update,
        )

    def create_bias_node(self, inputs: List[OpNode]) -> Bias:
        return self._builder.create_bias_node(inputs)

    def get_leaf(
        self, root_param: "Parameter", leaf_path: Tuple[str, ...]
    ) -> Optional["Parameter"]:
        return navigate_to_leaf(root_param, leaf_path)


class OpGraphBuilder:
    """Builder for operation graphs."""

    def __init__(self, model: "BuildModel") -> None:
        self.model = model
        self.node_counter = 0
        self.learned_nodes: Dict[Tuple[str, Tuple[str, ...]], LearnedValue] = {}

    def next_id(self, prefix: str) -> str:
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def create_bias_node(self, inputs: List[OpNode]) -> Bias:
        return Bias(id=self.next_id("bias"), inputs=inputs)

    def build_relationship_inputs(
        self,
        root_param: "Parameter",
        leaf_path: Tuple[str, ...],
        leaf: Optional["ScalarParameter"],
    ) -> List[OpNode]:
        relationships = collect_applicable_relationships(root_param, leaf_path, leaf)
        input_nodes: List[OpNode] = []
        for _, source_param, rel_type, outlier, source_subpath in relationships:
            input_node = self.build_source_node(
                source_param,
                source_subpath,
                leaf.units if leaf else None,
                rel_type,
                outlier,
                include_update=True,
            )
            if input_node is not None:
                input_nodes.append(input_node)
        return input_nodes

    def build_source_node(
        self,
        source_param: "Parameter",
        source_subpath: Tuple[str, ...],
        target_units: Optional[ParsedUnit],
        rel_type: Optional[Rel],
        outlier: Optional[OutlierHandling],
        *,
        include_update: bool,
    ) -> Optional[OpNode]:
        try:
            source_type, source_name, source_base_path = find_source_info(
                source_param, self.model
            )
        except ValueError:
            return None

        full_source_path = source_base_path + source_subpath

        if source_type == "sensor":
            source_root = self.model._sensors[source_name].parameter
        else:
            source_root = self.model._learned[source_name]

        source_leaf = navigate_to_leaf(source_root, full_source_path)
        source_units = source_leaf.units if source_leaf else None

        if source_type == "sensor":
            input_node: OpNode = SensorInput(
                id=self.next_id("sensor"),
                inputs=[],
                sensor_name=source_name,
                parameter_path=full_source_path,
                units=source_units,
            )
        else:
            learned_leaf = navigate_to_leaf(source_root, full_source_path)
            input_node = self.build_value_node(
                source_name,
                source_root,
                full_source_path,
                learned_leaf,
                is_output=False,
            )

        scale_factor = compute_scale_factor(source_units, target_units, rel_type)

        current_node = input_node

        if scale_factor != 1.0:
            scale_node = Scale(
                id=self.next_id("scale"),
                inputs=[current_node],
                scale_factor=scale_factor,
                source_units=source_units,
                target_units=target_units,
            )
            current_node = scale_node

        if rel_type in (Rel.delta, Rel.per_second):
            state_update_id = self.next_id("state_update")
            state_node = State(
                id=self.next_id("state"),
                inputs=[],
                state_name=state_update_id,
                initial_value=0.0,
                units=target_units,
            )

            state_update = StateUpdate(
                id=state_update_id,
                inputs=[state_node, current_node],
                state_ref=state_node,
                update_type=rel_type,
            )
            current_node = state_update

        if include_update:
            update_node = Update(
                id=self.next_id("update"),
                inputs=[current_node],
                outlier_handling=outlier,
            )
            return update_node

        return current_node

    def build_value_node(
        self,
        root_name: str,
        root_param: "Parameter",
        leaf_path: Tuple[str, ...],
        leaf: Optional["ScalarParameter"],
        *,
        is_output: bool,
    ) -> OpNode:
        if not is_output:
            cache_key = (root_name, leaf_path)
            if cache_key in self.learned_nodes:
                return self.learned_nodes[cache_key]

        helper = OpGraphHelper(self, root_name)
        custom_inputs = root_param.get_custom_ops(helper, root_name, leaf_path, leaf)

        if custom_inputs:
            input_nodes = custom_inputs
        else:
            input_nodes = self.build_relationship_inputs(root_param, leaf_path, leaf)

        if is_output:
            output_node = PredictionOutput(
                id=self.next_id("output"),
                inputs=input_nodes,
                parameter_name=root_name,
                path=leaf_path,
            )
            return output_node

        learned_node = LearnedValue(
            id=self.next_id("learned"),
            inputs=input_nodes,
            learned_name=root_name,
            parameter_path=leaf_path,
            units=leaf.units if leaf else None,
        )
        self.learned_nodes[(root_name, leaf_path)] = learned_node
        return learned_node

    def build(self) -> Dict[str, PredictionOutput]:
        outputs: Dict[str, PredictionOutput] = {}

        # Process each predicted parameter
        for pred_name, pred_param in self.model._predicted.items():
            # Find all scalar leaves
            scalar_leaves = collect_scalar_leaves(pred_param)

            for leaf_path, leaf in scalar_leaves:
                # Build the output path string
                full_path = (pred_name,) + leaf_path
                output_path_str = ".".join(full_path)

                output_node = self.build_value_node(
                    pred_name,
                    pred_param,
                    leaf_path,
                    leaf,
                    is_output=True,
                )
                outputs[output_path_str] = output_node

        return outputs


def render_op_graph_mermaid(outputs: Dict[str, PredictionOutput]) -> str:
    """Render an operation graph as a Mermaid flowchart.

    Args:
        outputs: Dict mapping output paths to PredictionOutput nodes.

    Returns:
        Mermaid flowchart text that can be rendered by Mermaid-compatible tools.
    """

    nodes: Dict[str, OpNode] = {}
    edges: set[Tuple[str, str]] = set()

    def format_path(path: Tuple[str, ...]) -> str:
        return ".".join(path) if path else ""

    def escape_label(label: str) -> str:
        return label.replace('"', '\\"')

    def node_label(node: OpNode) -> str:
        if isinstance(node, SensorInput):
            path = format_path(node.parameter_path)
            label = f"Sensor: {node.sensor_name}"
            if path:
                label += f".{path}"
            return label
        if isinstance(node, LearnedValue):
            path = format_path(node.parameter_path)
            label = f"Learned: {node.learned_name}"
            if path:
                label += f".{path}"
            return label
        if isinstance(node, Scale):
            return f"Scale x{node.scale_factor:g}"
        if isinstance(node, Bias):
            return "Bias (+)"
        if isinstance(node, State):
            return f"State: {node.state_name}"
        if isinstance(node, StateUpdate):
            if node.update_type:
                return f"State Update ({node.update_type.value})"
            return "State Update"
        if isinstance(node, Update):
            if node.outlier_handling:
                return f"Update ({node.outlier_handling.value})"
            return "Update"
        if isinstance(node, PredictionOutput):
            full_path = ".".join((node.parameter_name,) + node.path)
            return f"Output: {full_path}"
        return node.id

    def visit(node: OpNode) -> None:
        if node.id in nodes:
            return
        nodes[node.id] = node
        for input_node in node.inputs:
            edges.add((input_node.id, node.id))
            visit(input_node)

    for output_node in outputs.values():
        visit(output_node)

    lines = ["flowchart TD"]
    for node_id in sorted(nodes):
        label = escape_label(node_label(nodes[node_id]))
        lines.append(f'    {node_id}["{label}"]')
    for source_id, target_id in sorted(edges):
        lines.append(f"    {source_id} --> {target_id}")
    return "\n".join(lines)
