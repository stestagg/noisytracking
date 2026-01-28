"""Test the API structure from the README conceptual sketch."""

import pytest
import noisytracking as nt


def pose_from_matrix(x):
    """Mock function for sensor_data_format."""
    return {'position': x[:3], 'rotation': x[3:]}


def test_example():
    model = nt.setup(time_field="timestamp", time_units="sample")

    user_position = model.predicted(
        name="position",
        kind=nt.Motion(
            units={"length": "m", "angle": "rad"},
            expected_change=nt.CONSTANT_VELOCITY,
        ),
    )

    gps = model.sensor("gps", kind=nt.Position(units="m"))
    gps_heading = model.sensor("gps_heading", kind=nt.Rotation(units="rad"), sensor_data_format=lambda x: {'yaw': x})
    tracking = model.sensor(
        "tracking",
        kind=nt.Pose(units={"length": "m/sample", "angle": "rad/sample"}, ),
        sensor_data_format=pose_from_matrix,
    )

    # Absolute anchors (observation relationships)
    user_position.position.is_estimated_from(gps)
    user_position.rotation.yaw.is_estimated_from(gps_heading.yaw)

    # Learned calibration/bias for tracking (exposes .bias and .applied)
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

    # Integrate the corrected tracking stream into the predicted position
    user_position.is_estimated_from(tracking_bias.applied, rel=nt.Rel.delta, outlier_handling=nt.OutlierHandling.HEAVY_TAILED)

    with pytest.raises(NotImplementedError):
        _ = model.build()