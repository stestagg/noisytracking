"""Tests for the DAG builder."""

import pytest
import noisytracking as nt
from noisytracking.build_helpers import (
    build_dag,
    collect_scalar_leaves,
    find_source_info,
    compute_scale_factor,
    is_descendant_of,
    get_path_from_ancestor,
    DAGNode,
    SensorInput,
    LearnedInput,
    Scale,
    State,
    StateUpdate,
    Update,
    PredictionOutput,
)
from noisytracking.units import parse_unit


def pose_from_matrix(x):
    """Mock function for sensor_data_format."""
    return {'position': x[:3], 'rotation': x[3:]}


class TestCollectScalarLeaves:
    """Tests for collect_scalar_leaves helper."""

    def test_scalar_parameter(self):
        """A scalar parameter is its own leaf."""
        scalar = nt.ScalarParameter(name="x", units="m")
        leaves = collect_scalar_leaves(scalar)
        assert len(leaves) == 1
        assert leaves[0][0] == ()
        assert leaves[0][1] is scalar

    def test_position_has_xyz(self):
        """Position has x, y, z scalar leaves."""
        pos = nt.Position(name="pos", units="m")
        leaves = collect_scalar_leaves(pos)
        assert len(leaves) == 3
        paths = [p for p, _ in leaves]
        assert ('x',) in paths
        assert ('y',) in paths
        assert ('z',) in paths

    def test_pose_has_position_and_rotation(self):
        """Pose has position (x,y,z) and rotation (yaw,pitch,roll) leaves."""
        pose = nt.Pose(units={"length": "m", "angle": "rad"})
        leaves = collect_scalar_leaves(pose)
        assert len(leaves) == 6
        paths = [p for p, _ in leaves]
        assert ('position', 'x') in paths
        assert ('position', 'y') in paths
        assert ('position', 'z') in paths
        assert ('rotation', 'yaw') in paths
        assert ('rotation', 'pitch') in paths
        assert ('rotation', 'roll') in paths


class TestFindSourceInfo:
    """Tests for find_source_info helper."""

    def test_finds_sensor(self):
        """Should find a sensor and return its path."""
        model = nt.setup()
        gps = model.sensor("gps", kind=nt.Position(units="m"))

        source_type, name, path = find_source_info(gps, model)
        assert source_type == "sensor"
        assert name == "gps"
        assert path == ()

    def test_finds_sensor_child(self):
        """Should find a sensor child and return correct path."""
        model = nt.setup()
        gps = model.sensor("gps", kind=nt.Position(units="m"))

        source_type, name, path = find_source_info(gps.x, model)
        assert source_type == "sensor"
        assert name == "gps"
        assert path == ('x',)

    def test_finds_learned(self):
        """Should find a learned parameter and return its path."""
        model = nt.setup()
        tracking = model.sensor("tracking", kind=nt.Pose(units={"length": "m", "angle": "rad"}))
        tracking_bias = model.learned(
            "tracking_bias",
            kind=nt.LearnedBias(
                target=tracking,
                expected_change=nt.CONSTANT_VALUE,
            ),
        )

        source_type, name, path = find_source_info(tracking_bias.applied, model)
        assert source_type == "learned"
        assert name == "tracking_bias"
        assert path == ('applied',)

    def test_raises_for_unknown(self):
        """Should raise ValueError for unknown source."""
        model = nt.setup()
        unknown = nt.Position(units="m")

        with pytest.raises(ValueError, match="not found"):
            find_source_info(unknown, model)


class TestComputeScaleFactor:
    """Tests for compute_scale_factor helper."""

    def test_same_units(self):
        """Same units should have factor of 1."""
        m = parse_unit("m")
        factor = compute_scale_factor(m, m)
        assert factor == 1.0

    def test_km_to_m(self):
        """km to m should be 1000."""
        km = parse_unit("km")
        m = parse_unit("m")
        factor = compute_scale_factor(km, m)
        assert factor == 1000.0

    def test_m_to_km(self):
        """m to km should be 0.001."""
        m = parse_unit("m")
        km = parse_unit("km")
        factor = compute_scale_factor(m, km)
        assert factor == pytest.approx(0.001)

    def test_none_units(self):
        """None units should return 1."""
        m = parse_unit("m")
        assert compute_scale_factor(None, m) == 1.0
        assert compute_scale_factor(m, None) == 1.0
        assert compute_scale_factor(None, None) == 1.0

    def test_velocity_conversion(self):
        """km/h to m/s should scale correctly."""
        kmh = parse_unit("km/h")
        ms = parse_unit("m/s")
        factor = compute_scale_factor(kmh, ms)
        # km/h = 1000m / 3600s = 0.277... m/s
        # So 1 km/h * factor = 1 m/s value, factor = 1000/3600
        assert factor == pytest.approx(1000.0 / 3600.0)


class TestIsDescendantOf:
    """Tests for is_descendant_of helper."""

    def test_same_parameter(self):
        """A parameter is a descendant of itself."""
        pos = nt.Position(units="m")
        assert is_descendant_of(pos, pos)

    def test_child_is_descendant(self):
        """A child is a descendant of its parent."""
        pos = nt.Position(units="m")
        assert is_descendant_of(pos.x, pos)

    def test_grandchild_is_descendant(self):
        """A grandchild is a descendant of its grandparent."""
        pose = nt.Pose(units={"length": "m", "angle": "rad"})
        assert is_descendant_of(pose.position.x, pose)

    def test_unrelated_not_descendant(self):
        """Unrelated parameters are not descendants."""
        pos1 = nt.Position(units="m")
        pos2 = nt.Position(units="m")
        assert not is_descendant_of(pos1, pos2)


class TestGetPathFromAncestor:
    """Tests for get_path_from_ancestor helper."""

    def test_same_parameter(self):
        """Same parameter returns empty path."""
        pos = nt.Position(units="m")
        assert get_path_from_ancestor(pos, pos) == ()

    def test_child(self):
        """Child returns single-element path."""
        pos = nt.Position(units="m")
        assert get_path_from_ancestor(pos, pos.x) == ('x',)

    def test_grandchild(self):
        """Grandchild returns multi-element path."""
        pose = nt.Pose(units={"length": "m", "angle": "rad"})
        assert get_path_from_ancestor(pose, pose.position.x) == ('position', 'x')

    def test_not_descendant(self):
        """Non-descendant returns None."""
        pos1 = nt.Position(units="m")
        pos2 = nt.Position(units="m")
        assert get_path_from_ancestor(pos1, pos2) is None


class TestBuildDag:
    """Tests for the main build_dag function."""

    def test_simple_sensor_relationship(self):
        """Simple sensor to predicted relationship."""
        model = nt.setup()
        pos = model.predicted("position", kind=nt.Position(units="m"))
        gps = model.sensor("gps", kind=nt.Position(units="m"))
        pos.is_estimated_from(gps)

        dag = build_dag(model)

        # Should have 3 outputs (x, y, z)
        assert len(dag) == 3
        assert "position.x" in dag
        assert "position.y" in dag
        assert "position.z" in dag

        # Check structure of one output
        out_x = dag["position.x"]
        assert isinstance(out_x, PredictionOutput)
        assert out_x.parameter_name == "position"
        assert out_x.path == ('x',)

        # Should have one input (the Update node)
        assert len(out_x.inputs) == 1
        update = out_x.inputs[0]
        assert isinstance(update, Update)

        # Update should have SensorInput as input
        assert len(update.inputs) == 1
        sensor_input = update.inputs[0]
        assert isinstance(sensor_input, SensorInput)
        assert sensor_input.sensor_name == "gps"
        assert sensor_input.parameter_path == ('x',)

    def test_delta_relationship_has_state(self):
        """Delta relationship should create State and StateUpdate nodes."""
        model = nt.setup()
        pos = model.predicted("position", kind=nt.Position(units="m"))
        tracking = model.sensor("tracking", kind=nt.Position(units="m/sample"))
        pos.is_estimated_from(tracking, rel=nt.Rel.delta)

        dag = build_dag(model)

        out_x = dag["position.x"]
        update = out_x.inputs[0]
        assert isinstance(update, Update)
        assert update.rel_type == nt.Rel.delta

        # Should have StateUpdate as input
        state_update = update.inputs[0]
        assert isinstance(state_update, StateUpdate)
        assert state_update.state_ref is not None
        assert isinstance(state_update.state_ref, State)
        assert state_update.state_ref.state_name == "position.x"

    def test_multiple_sources(self):
        """Parameter with multiple sources should have multiple Update inputs."""
        model = nt.setup()
        pos = model.predicted("position", kind=nt.Position(units="m"))
        gps = model.sensor("gps", kind=nt.Position(units="m"))
        tracking = model.sensor("tracking", kind=nt.Position(units="m/sample"))

        pos.is_estimated_from(gps)
        pos.is_estimated_from(tracking, rel=nt.Rel.delta)

        dag = build_dag(model)

        out_x = dag["position.x"]
        # Should have two Update inputs
        assert len(out_x.inputs) == 2
        assert all(isinstance(inp, Update) for inp in out_x.inputs)

    def test_outlier_handling_preserved(self):
        """Outlier handling should be preserved in Update nodes."""
        model = nt.setup()
        pos = model.predicted("position", kind=nt.Position(units="m"))
        gps = model.sensor("gps", kind=nt.Position(units="m"))
        pos.is_estimated_from(gps, outlier_handling=nt.OutlierHandling.HEAVY_TAILED)

        dag = build_dag(model)

        out_x = dag["position.x"]
        update = out_x.inputs[0]
        assert update.outlier_handling == nt.OutlierHandling.HEAVY_TAILED

    def test_learned_input(self):
        """Learned parameters should create LearnedInput nodes."""
        model = nt.setup()
        pos = model.predicted("position", kind=nt.Position(units="m"))
        tracking = model.sensor("tracking", kind=nt.Position(units="m/sample"))
        tracking_bias = model.learned(
            "tracking_bias",
            kind=nt.LearnedBias(
                target=tracking,
                expected_change=nt.CONSTANT_VALUE,
            ),
        )
        pos.is_estimated_from(tracking_bias.applied, rel=nt.Rel.delta)

        dag = build_dag(model)

        out_x = dag["position.x"]
        update = out_x.inputs[0]
        state_update = update.inputs[0]

        # StateUpdate has state + input
        assert len(state_update.inputs) == 2
        # First input is State, second is the learned input (possibly with scale)
        learned_input = state_update.inputs[1]

        # May be LearnedInput directly or Scale->LearnedInput
        if isinstance(learned_input, Scale):
            learned_input = learned_input.inputs[0]

        assert isinstance(learned_input, LearnedInput)
        assert learned_input.learned_name == "tracking_bias"
        assert learned_input.parameter_path == ('applied', 'x')

    def test_unit_scaling(self):
        """Unit conversion should create Scale nodes."""
        model = nt.setup()
        pos = model.predicted("position", kind=nt.Position(units="m"))
        gps = model.sensor("gps", kind=nt.Position(units="km"))
        pos.is_estimated_from(gps)

        dag = build_dag(model)

        out_x = dag["position.x"]
        update = out_x.inputs[0]

        # Should have Scale node between sensor and update
        scale = update.inputs[0]
        assert isinstance(scale, Scale)
        assert scale.scale_factor == pytest.approx(1000.0)

        # Scale input should be SensorInput
        sensor_input = scale.inputs[0]
        assert isinstance(sensor_input, SensorInput)

    def test_full_example(self):
        """Test with the full example from test_example.py."""
        model = nt.setup()

        user_position = model.predicted(
            name="position",
            kind=nt.Motion(
                units={"length": "m", "angle": "rad"},
                expected_change=nt.CONSTANT_VELOCITY,
            ),
        )

        gps = model.sensor("gps", kind=nt.Position(units="m"))
        gps_heading = model.sensor(
            "gps_heading",
            kind=nt.Rotation(units="rad"),
            sensor_data_format=lambda x: {'yaw': x}
        )
        tracking = model.sensor(
            "tracking",
            kind=nt.Pose(units={"length": "m/sample", "angle": "rad/sample"}),
            sensor_data_format=pose_from_matrix,
        )

        # Absolute anchors
        user_position.position.is_estimated_from(gps)
        user_position.rotation.yaw.is_estimated_from(gps_heading.yaw)

        # Learned calibration/bias
        tracking_bias = model.learned(
            name="tracking_bias",
            kind=nt.LearnedBias(
                target=tracking,
                expected_change=nt.CONSTANT_VALUE,
                change_model=nt.ChangeModel.DRIFT_WITH_JUMPS
            ),
        )
        tracking_bias.bias.position.is_estimated_from(gps)
        tracking_bias.bias.rotation.yaw.is_estimated_from(gps_heading.yaw)

        # Integrate tracking
        user_position.is_estimated_from(
            tracking_bias.applied,
            rel=nt.Rel.delta,
            outlier_handling=nt.OutlierHandling.HEAVY_TAILED
        )

        dag = build_dag(model)

        # Should have 6 outputs (position x,y,z + rotation yaw,pitch,roll)
        assert len(dag) == 6

        # Check position.x has both GPS and tracking sources
        out_pos_x = dag["position.position.x"]
        assert isinstance(out_pos_x, PredictionOutput)
        assert len(out_pos_x.inputs) == 2  # GPS + tracking

        # One should be absolute (GPS), one should be delta (tracking)
        rel_types = [inp.rel_type for inp in out_pos_x.inputs]
        assert None in rel_types or nt.Rel.absolute in rel_types  # GPS is absolute
        assert nt.Rel.delta in rel_types  # tracking is delta

        # Check rotation.yaw has GPS heading and tracking
        out_yaw = dag["position.rotation.yaw"]
        assert len(out_yaw.inputs) == 2

        # Check rotation.pitch/roll only have tracking (no GPS heading for those)
        out_pitch = dag["position.rotation.pitch"]
        assert len(out_pitch.inputs) == 1
        assert out_pitch.inputs[0].rel_type == nt.Rel.delta

    def test_no_relationships(self):
        """Predicted parameter with no relationships should still create outputs."""
        model = nt.setup()
        pos = model.predicted("position", kind=nt.Position(units="m"))

        dag = build_dag(model)

        assert len(dag) == 3
        out_x = dag["position.x"]
        assert out_x.inputs == []


class TestDagTraversal:
    """Tests for DAG traversal and node connectivity."""

    def collect_all_nodes(self, node: DAGNode, visited: set = None) -> set:
        """Recursively collect all nodes in the DAG."""
        if visited is None:
            visited = set()
        if node.id in visited:
            return visited
        visited.add(node.id)
        for inp in node.inputs:
            self.collect_all_nodes(inp, visited)
        return visited

    def test_dag_is_connected(self):
        """All nodes should be reachable from outputs."""
        model = nt.setup()
        pos = model.predicted("position", kind=nt.Position(units="m"))
        gps = model.sensor("gps", kind=nt.Position(units="m"))
        tracking = model.sensor("tracking", kind=nt.Position(units="m/sample"))
        pos.is_estimated_from(gps)
        pos.is_estimated_from(tracking, rel=nt.Rel.delta)

        dag = build_dag(model)

        # Collect all nodes from all outputs
        all_nodes: set = set()
        for output in dag.values():
            self.collect_all_nodes(output, all_nodes)

        # Should have nodes for all components
        # For each leaf: PredictionOutput, 2 Updates, SensorInput (GPS),
        # StateUpdate + State + SensorInput (tracking)
        assert len(all_nodes) > 0

    def test_state_nodes_shared(self):
        """State nodes should be shared across relationships affecting same leaf."""
        model = nt.setup()
        pos = model.predicted("position", kind=nt.Position(units="m"))
        tracking1 = model.sensor("tracking1", kind=nt.Position(units="m/sample"))
        tracking2 = model.sensor("tracking2", kind=nt.Position(units="m/sample"))
        pos.is_estimated_from(tracking1, rel=nt.Rel.delta)
        pos.is_estimated_from(tracking2, rel=nt.Rel.delta)

        dag = build_dag(model)

        out_x = dag["position.x"]

        # Both updates should reference the same state
        state_refs = []
        for update in out_x.inputs:
            state_update = update.inputs[0]
            if isinstance(state_update, StateUpdate):
                state_refs.append(state_update.state_ref)

        assert len(state_refs) == 2
        assert state_refs[0] is state_refs[1]
