# noisytracking

A Python library for noisytracking

##  Conceptual sketch:

```
import noisytracking as nt
# Some things in nt:
# Dynamics "order" (single source of truth)
# CONSTANT_VALUE        = 0  # ORDER0: value persists
# CONSTANT_RATE         = 1  # ORDER1: first derivative persists
# CONSTANT_CURVATURE    = 2  # ORDER2: second derivative persists

# Friendly aliases (pure ergonomics; map to the same integers)
# STEADY_STATE          = CONSTANT_RATE

# Optional motion-flavoured aliases (still just ints)
# CONSTANT_POSITION     = CONSTANT_VALUE
# CONSTANT_VELOCITY     = CONSTANT_RATE
# CONSTANT_ACCELERATION = CONSTANT_CURVATURE


# class OutlierHandling(Enum):
#     NONE        = "none"         # plain residual (Gaussian)
#     GATED       = "gated"        # reject/ignore large residuals
#     HEAVY_TAILED= "heavy_tailed" # down-weight outliers smoothly
#     MIXTURE     = "mixture"      # explicit inlier/outlier mixture


model = nt.setup(time_field="timestamp", time_units="sample", sample_time_policy='sequential_buckets')

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

predictor = model.build()

...

predictor.update_from_sensor(timestamp=1234567890.0, gps={'x': 52.12343456, 'y': 2.012345, 'z': 0.0})
predictor.update_from_sensor(timestamp=1234567890.0, gps_heading=0.5, tracking=np.array([[...]]))

print(predictor.predict(timestamp=1234567892).position.position)
```

## Architecture Notes

Units module will come later (nt.units.*), but there are some concepts:
 - units dimension (one of length/time/angle/temperature etc..)
 - units are parsed into a structure that models them so we know that m/s is <meters +1>, <seconds -1>, and we can work out relationships by integration/differentiation between them.
 - every unit belongs to a dimension, and all units can convert values to any other unit in the same  dimension.
 - units can be: defined directly during build, OR defined by dimension (i.e. {'legnth': 'cm'}) or left as default values.  Compound parameters pass units to children, but this raises errors on incompatibility (i.e. Pose(units='cm') errors because Pose.rotation has the 'angle' dimension. but Pose(units={'length': 'cm'})) results in Pose.rotation units being default, while Pose.position units = cm.

every 'variable' in the model builder should be a Parameter.  A parameter is one of:
1. ScalarParameter (floating point variable that will be built into a prior/variable/estimated in the predictor) - has a unit (defaults to sensible default ('m'/'m/s' etc.))
2. 'CompoundParameter', has one or more sub-parameters each of which are named and are scalar or compound parameters.

Parameters have a child_parameters property that returns a dict of child parameters for any parameter (scalar returns empty dict)

Two CompoundParameters can directly interact if the units and names of their parameters match.  (i.e. nt.Motion.position and nt.Position, but not nt.Motion.rotation and nt.Position) and the interaction is either handled directly, or broadcast to the matching child parameters depending on the operation.


## Installation

```bash
uv pip install -e .
```

## Development

Install with development dependencies:

```bash
uv pip install -e ".[dev]"
```

## Testing

Run tests with pytest:

```bash
pytest
```
\