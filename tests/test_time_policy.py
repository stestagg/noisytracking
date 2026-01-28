"""Tests for time policy classes."""

import pytest
import noisytracking as nt
from noisytracking.time_policy import (
    DuplicateValueError,
    Observations,
    SequentialBucketsPolicy,
    StaleTimestampError,
    TimePolicy,
    TimePolicyError,
    create_time_policy,
)


class TestSequentialBucketsPolicy:
    """Tests for SequentialBucketsPolicy."""

    def test_basic_add_and_retrieve(self):
        """Test basic adding and retrieving of samples."""
        policy = SequentialBucketsPolicy()

        policy.add_sample(1.0, "sensor1", {"x": 1.0, "y": 2.0})
        policy.add_sample(2.0, "sensor1", {"x": 1.1, "y": 2.1})

        samples = policy.get_samples_up_to(2.0)

        assert len(samples) == 2
        assert samples[0] == Observations(
            timestamp=1.0,
            previous_timestamp=None,
            sensor_data={"sensor1": {"x": 1.0, "y": 2.0}},
        )
        assert samples[1] == Observations(
            timestamp=2.0,
            previous_timestamp=1.0,
            sensor_data={"sensor1": {"x": 1.1, "y": 2.1}},
        )

    def test_multiple_sensors_same_timestamp(self):
        """Test multiple sensors at the same timestamp."""
        policy = SequentialBucketsPolicy()

        policy.add_sample(1.0, "sensor1", {"x": 1.0, "y": 2.0})
        policy.add_sample(1.0, "sensor2", {"yaw": 0.5})

        samples = policy.get_samples_up_to(1.0)

        assert len(samples) == 1
        assert samples[0] == Observations(
            timestamp=1.0,
            previous_timestamp=None,
            sensor_data={"sensor1": {"x": 1.0, "y": 2.0}, "sensor2": {"yaw": 0.5}},
        )

    def test_samples_returned_in_timestamp_order(self):
        """Test that samples are returned sorted by timestamp."""
        policy = SequentialBucketsPolicy()

        # Add in non-sorted order (different sensors can have different timestamps)
        policy.add_sample(3.0, "sensor2", {"z": 3.0})
        policy.add_sample(1.0, "sensor1", {"x": 1.0})
        policy.add_sample(2.0, "sensor1", {"x": 2.0})

        samples = policy.get_samples_up_to(3.0)

        timestamps = [s.timestamp for s in samples]
        assert timestamps == [1.0, 2.0, 3.0]

    def test_get_samples_up_to_partial(self):
        """Test getting only samples up to a certain timestamp."""
        policy = SequentialBucketsPolicy()

        policy.add_sample(1.0, "sensor1", {"x": 1.0})
        policy.add_sample(2.0, "sensor1", {"x": 2.0})
        policy.add_sample(3.0, "sensor1", {"x": 3.0})

        samples = policy.get_samples_up_to(2.0)

        assert len(samples) == 2
        assert samples[0].timestamp == 1.0
        assert samples[1].timestamp == 2.0

        # Remaining sample should still be there
        remaining = policy.get_samples_up_to(3.0)
        assert len(remaining) == 1
        assert remaining[0].timestamp == 3.0

    def test_same_timestamp_allowed_for_same_sensor(self):
        """Test that the same timestamp is allowed if data is identical."""
        policy = SequentialBucketsPolicy()

        policy.add_sample(1.0, "sensor1", {"x": 1.0})
        # Adding identical data at same timestamp should be OK
        policy.add_sample(1.0, "sensor1", {"x": 1.0})

        samples = policy.get_samples_up_to(1.0)
        assert len(samples) == 1

    def test_different_sensors_independent_monotonicity(self):
        """Test that different sensors have independent timestamp tracking."""
        policy = SequentialBucketsPolicy()

        policy.add_sample(2.0, "sensor1", {"x": 2.0})
        # sensor2 can have an earlier timestamp than sensor1's last timestamp
        policy.add_sample(1.0, "sensor2", {"y": 1.0})

        samples = policy.get_samples_up_to(2.0)
        assert len(samples) == 2

    def test_stale_timestamp_rejected_after_request(self):
        """Test that stale timestamps are rejected after a model request."""
        policy = SequentialBucketsPolicy()

        policy.add_sample(1.0, "sensor1", {"x": 1.0})
        policy.get_samples_up_to(2.0)  # Sets last request timestamp to 2.0

        with pytest.raises(StaleTimestampError) as exc_info:
            policy.add_sample(1.5, "sensor1", {"x": 1.5})

        assert "1.5" in str(exc_info.value)
        assert "2.0" in str(exc_info.value)

    def test_stale_timestamp_ok_before_first_request(self):
        """Test that any timestamp is OK before the first model request."""
        policy = SequentialBucketsPolicy()

        # No request yet, so any timestamp order is OK for different sensors
        policy.add_sample(5.0, "sensor1", {"x": 5.0})
        policy.add_sample(1.0, "sensor2", {"y": 1.0})

        samples = policy.get_samples_up_to(5.0)
        assert len(samples) == 2

    def test_duplicate_value_error_different_data(self):
        """Test that conflicting data at same timestamp raises error."""
        policy = SequentialBucketsPolicy()

        policy.add_sample(1.0, "sensor1", {"x": 1.0})

        with pytest.raises(DuplicateValueError) as exc_info:
            policy.add_sample(1.0, "sensor1", {"x": 2.0})

        assert "sensor1" in str(exc_info.value)
        assert "1.0" in str(exc_info.value)

    def test_clear_resets_all_state(self):
        """Test that clear() resets all internal state."""
        policy = SequentialBucketsPolicy()

        policy.add_sample(1.0, "sensor1", {"x": 1.0})
        policy.get_samples_up_to(2.0)  # Sets last request timestamp

        policy.clear()

        # Should be able to add samples at any timestamp now
        policy.add_sample(0.5, "sensor1", {"x": 0.5})

        samples = policy.get_samples_up_to(1.0)
        assert len(samples) == 1
        assert samples[0] == Observations(
            timestamp=0.5,
            previous_timestamp=None,
            sensor_data={"sensor1": {"x": 0.5}},
        )

    def test_empty_get_samples(self):
        """Test getting samples when none exist."""
        policy = SequentialBucketsPolicy()

        samples = policy.get_samples_up_to(1.0)
        assert samples == []

    def test_get_samples_removes_returned_samples(self):
        """Test that get_samples_up_to removes the returned samples."""
        policy = SequentialBucketsPolicy()

        policy.add_sample(1.0, "sensor1", {"x": 1.0})

        samples1 = policy.get_samples_up_to(1.0)
        assert len(samples1) == 1

        samples2 = policy.get_samples_up_to(1.0)
        assert len(samples2) == 0


class TestCreateTimePolicy:
    """Tests for the create_time_policy factory function."""

    def test_create_sequential_buckets(self):
        """Test creating a sequential_buckets policy."""
        policy = create_time_policy("sequential_buckets")
        assert isinstance(policy, SequentialBucketsPolicy)
        assert isinstance(policy, TimePolicy)

    def test_unknown_policy_raises_error(self):
        """Test that unknown policy names raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_time_policy("unknown_policy")

        assert "unknown_policy" in str(exc_info.value)
        assert "sequential_buckets" in str(exc_info.value)


class TestBuilderIntegration:
    """Tests for Builder integration with TimePolicy."""

    def test_builder_creates_time_policy(self):
        """Test that Builder creates a time policy instance."""
        builder = nt.setup(
            sample_time_policy=nt.SequentialBucketsPolicy(),
        )

        assert hasattr(builder, "time_policy")
        assert isinstance(builder.time_policy, nt.TimePolicy)
        assert isinstance(builder.time_policy, nt.SequentialBucketsPolicy)

    def test_builder_time_policy_is_usable(self):
        """Test that the builder's time policy can be used."""
        builder = nt.setup(
            sample_time_policy=nt.SequentialBucketsPolicy(),
        )

        builder.time_policy.add_sample(1.0, "gps", {"x": 1.0, "y": 2.0})
        samples = builder.time_policy.get_samples_up_to(1.0)

        assert len(samples) == 1
        assert samples[0] == Observations(
            timestamp=1.0,
            previous_timestamp=None,
            sensor_data={"gps": {"x": 1.0, "y": 2.0}},
        )
