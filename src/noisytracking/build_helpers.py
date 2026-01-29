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
class LearnedInput(DAGNode):
    """Input node representing a learned parameter."""
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


@dataclass
class Update(DAGNode):
    """Bayesian update / measurement fusion."""
    outlier_handling: Optional[OutlierHandling] = None
    rel_type: Optional[Rel] = None


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

    # Track state nodes by path to reuse them
    state_nodes: Dict[str, State] = {}

    # Process each predicted parameter
    for pred_name, pred_param in model._predicted.items():
        # Find all scalar leaves
        scalar_leaves = collect_scalar_leaves(pred_param)

        for leaf_path, leaf in scalar_leaves:
            # Build the output path string
            full_path = (pred_name,) + leaf_path
            output_path_str = ".".join(full_path)

            # Collect all applicable relationships for this leaf
            relationships = collect_applicable_relationships(pred_param, leaf_path, leaf)

            if not relationships:
                # No relationships - just create output node with no inputs
                output_node = PredictionOutput(
                    id=next_id("output"),
                    inputs=[],
                    parameter_name=pred_name,
                    path=leaf_path,
                )
                outputs[output_path_str] = output_node
                continue

            # Build input nodes for each relationship
            input_nodes: List[DAGNode] = []

            for target_param, source_param, rel_type, outlier, source_subpath in relationships:
                # Find source info (sensor or learned)
                try:
                    source_type, source_name, source_base_path = find_source_info(
                        source_param, model
                    )
                except ValueError:
                    # Source not found in model - skip this relationship
                    continue

                # The full path within the source is base_path + subpath
                full_source_path = source_base_path + source_subpath

                # Navigate to the source leaf to get its units
                if source_type == "sensor":
                    source_root = model._sensors[source_name].parameter
                else:
                    source_root = model._learned[source_name]

                source_leaf = navigate_to_leaf(source_root, full_source_path)
                source_units = source_leaf.units if source_leaf else None

                # Create input node
                if source_type == "sensor":
                    input_node: DAGNode = SensorInput(
                        id=next_id("sensor"),
                        inputs=[],
                        sensor_name=source_name,
                        parameter_path=full_source_path,
                        units=source_units,
                    )
                else:
                    input_node = LearnedInput(
                        id=next_id("learned"),
                        inputs=[],
                        learned_name=source_name,
                        parameter_path=full_source_path,
                        units=source_units,
                    )

                # Check if we need a scale node
                target_units = leaf.units
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

                # Handle delta/per_second relationships with State/StateUpdate
                if rel_type in (Rel.delta, Rel.per_second):
                    # Get or create state node for this leaf
                    state_key = output_path_str
                    if state_key not in state_nodes:
                        state_node = State(
                            id=next_id("state"),
                            inputs=[],
                            state_name=output_path_str,
                            initial_value=0.0,
                            units=target_units,
                        )
                        state_nodes[state_key] = state_node
                    else:
                        state_node = state_nodes[state_key]

                    # Create StateUpdate node
                    state_update = StateUpdate(
                        id=next_id("state_update"),
                        inputs=[state_node, current_node],
                        state_ref=state_node,
                    )
                    current_node = state_update

                # Create Update node for this relationship
                update_node = Update(
                    id=next_id("update"),
                    inputs=[current_node],
                    outlier_handling=outlier,
                    rel_type=rel_type,
                )

                input_nodes.append(update_node)

            # Create the output node combining all inputs
            output_node = PredictionOutput(
                id=next_id("output"),
                inputs=input_nodes,
                parameter_name=pred_name,
                path=leaf_path,
            )
            outputs[output_path_str] = output_node

    return outputs
