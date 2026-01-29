"""DAG builder for BuildModel."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from .constants import OutlierHandling, Rel
from .units import ParsedUnit, Exponent

if TYPE_CHECKING:
    from .builder import BuildModel
    from .parameter import Parameter, ScalarParameter


# ============================================================================
# DAG Node Classes
# ============================================================================


@dataclass
class DAGNode(ABC):
    """Base class for all DAG nodes."""
    id: str
    inputs: List["DAGNode"] = field(default_factory=list)


# Input Nodes


@dataclass
class SensorInput(DAGNode):
    """Input node representing a sensor measurement."""
    sensor_name: str = ""
    parameter_path: Tuple[str, ...] = ()
    units: Optional[ParsedUnit] = None


@dataclass
class LearnedValue(DAGNode):
    """Value node representing a learned parameter."""
    learned_name: str = ""
    parameter_path: Tuple[str, ...] = ()
    units: Optional[ParsedUnit] = None


# Operation Nodes


@dataclass
class Scale(DAGNode):
    """Scales input by a factor (for unit conversion)."""
    scale_factor: float = 1.0
    source_units: Optional[ParsedUnit] = None
    target_units: Optional[ParsedUnit] = None


@dataclass
class State(DAGNode):
    """Holds accumulated state for delta/per_second relationships."""
    state_name: str = ""
    initial_value: float = 0.0
    units: Optional[ParsedUnit] = None


@dataclass
class StateUpdate(DAGNode):
    """Updates state: new_value = state + scaled_delta."""
    state_ref: Optional[State] = None
    update_type: Optional[Rel] = None


@dataclass
class Update(DAGNode):
    """Bayesian update / measurement fusion."""
    outlier_handling: Optional[OutlierHandling] = None


# Output Nodes


@dataclass
class PredictionOutput(DAGNode):
    """Output node for a predicted scalar parameter."""
    parameter_name: str = ""
    path: Tuple[str, ...] = ()


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
    leaf: "ScalarParameter",
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
    results: List[Tuple["Parameter", "Parameter", Optional[Rel], Optional[OutlierHandling], Tuple[str, ...]]] = []

    # Walk the path, collecting relationships from each parameter
    current = root
    for i, name in enumerate(leaf_path + (None,)):  # type: ignore
        # Get relationships at this level
        for rel_obj in current._relationships:
            # For relationships at ancestor level, we need the remaining path
            remaining_path = leaf_path[i:] if name is not None else ()
            results.append((
                current,
                rel_obj.source,
                rel_obj.rel,
                rel_obj.outlier_handling,
                remaining_path,
            ))

        # Navigate to next child
        if name is not None:
            children = current.child_parameters
            if name in children:
                current = children[name]
            else:
                break

    return results


# ============================================================================
# Main DAG Builder
# ============================================================================


def build_dag(model: "BuildModel") -> Dict[str, PredictionOutput]:
    """Build a DAG from a BuildModel.

    For each predicted parameter, traces from scalar leaf parameters backwards
    through relationships to sensors and learned parameters, creating operation
    nodes along the way.

    Args:
        model: The BuildModel containing predicted, sensor, and learned parameters.

    Returns:
        Dict mapping output paths (e.g. "position.position.x") to PredictionOutput nodes.
        The full DAG is traversable from these terminal nodes.
    """
    outputs: Dict[str, PredictionOutput] = {}
    node_counter = 0

    def next_id(prefix: str) -> str:
        nonlocal node_counter
        node_counter += 1
        return f"{prefix}_{node_counter}"

    learned_nodes: Dict[Tuple[str, Tuple[str, ...]], LearnedValue] = {}

    def build_value_node(
        root_name: str,
        root_param: "Parameter",
        leaf_path: Tuple[str, ...],
        leaf: Optional["ScalarParameter"],
        *,
        is_output: bool,
    ) -> DAGNode:
        if not is_output:
            cache_key = (root_name, leaf_path)
            if cache_key in learned_nodes:
                return learned_nodes[cache_key]

        relationships = collect_applicable_relationships(root_param, leaf_path, leaf)
        if not relationships:
            try:
                from .learned import LearnedBias
            except ImportError:
                LearnedBias = None  # type: ignore[assignment]
            if LearnedBias and isinstance(root_param, LearnedBias):
                if leaf_path and leaf_path[0] == "applied":
                    source_subpath = leaf_path[1:]
                    relationships = [
                        (root_param, root_param.target, None, None, source_subpath),
                        (root_param, root_param.bias, None, None, source_subpath),
                    ]
        input_nodes: List[DAGNode] = []

        for _, source_param, rel_type, outlier, source_subpath in relationships:
            try:
                source_type, source_name, source_base_path = find_source_info(
                    source_param, model
                )
            except ValueError:
                continue

            full_source_path = source_base_path + source_subpath

            if source_type == "sensor":
                source_root = model._sensors[source_name].parameter
            else:
                source_root = model._learned[source_name]

            source_leaf = navigate_to_leaf(source_root, full_source_path)
            source_units = source_leaf.units if source_leaf else None

            if source_type == "sensor":
                input_node: DAGNode = SensorInput(
                    id=next_id("sensor"),
                    inputs=[],
                    sensor_name=source_name,
                    parameter_path=full_source_path,
                    units=source_units,
                )
            else:
                learned_leaf = navigate_to_leaf(source_root, full_source_path)
                input_node = build_value_node(
                    source_name,
                    source_root,
                    full_source_path,
                    learned_leaf,
                    is_output=False,
                )

            target_units = leaf.units if leaf else None
            scale_factor = compute_scale_factor(source_units, target_units, rel_type)

            current_node = input_node

            if scale_factor != 1.0:
                scale_node = Scale(
                    id=next_id("scale"),
                    inputs=[current_node],
                    scale_factor=scale_factor,
                    source_units=source_units,
                    target_units=target_units,
                )
                current_node = scale_node

            if rel_type in (Rel.delta, Rel.per_second):
                state_update_id = next_id("state_update")
                state_node = State(
                    id=next_id("state"),
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

            update_node = Update(
                id=next_id("update"),
                inputs=[current_node],
                outlier_handling=outlier,
            )

            input_nodes.append(update_node)

        if is_output:
            output_node = PredictionOutput(
                id=next_id("output"),
                inputs=input_nodes,
                parameter_name=root_name,
                path=leaf_path,
            )
            return output_node

        learned_node = LearnedValue(
            id=next_id("learned"),
            inputs=input_nodes,
            learned_name=root_name,
            parameter_path=leaf_path,
            units=leaf.units if leaf else None,
        )
        learned_nodes[(root_name, leaf_path)] = learned_node
        return learned_node

    # Process each predicted parameter
    for pred_name, pred_param in model._predicted.items():
        # Find all scalar leaves
        scalar_leaves = collect_scalar_leaves(pred_param)

        for leaf_path, leaf in scalar_leaves:
            # Build the output path string
            full_path = (pred_name,) + leaf_path
            output_path_str = ".".join(full_path)

            output_node = build_value_node(
                pred_name,
                pred_param,
                leaf_path,
                leaf,
                is_output=True,
            )
            outputs[output_path_str] = output_node

    return outputs


def render_dag_mermaid(outputs: Dict[str, PredictionOutput]) -> str:
    """Render a DAG as a Mermaid flowchart.

    Args:
        outputs: Dict mapping output paths to PredictionOutput nodes.

    Returns:
        Mermaid flowchart text that can be rendered by Mermaid-compatible tools.
    """
    nodes: Dict[str, DAGNode] = {}
    edges: set[Tuple[str, str]] = set()

    def format_path(path: Tuple[str, ...]) -> str:
        return ".".join(path) if path else ""

    def escape_label(label: str) -> str:
        return label.replace('"', '\\"')

    def node_label(node: DAGNode) -> str:
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

    def visit(node: DAGNode) -> None:
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
