"""Time policy classes for buffering and aggregating sensor samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class TimePolicyError(Exception):
    """Base exception for time policy errors."""

    pass


class DuplicateValueError(TimePolicyError):
    """Raised when conflicting values are provided at the same timestamp."""

    pass


class StaleTimestampError(TimePolicyError):
    """Raised when a timestamp is older than the last model request."""

    pass


class TimePolicy(ABC):
    """Abstract base class for time policies.

    Time policies buffer and aggregate sensor samples before they're consumed
    by the model.
    """

    @abstractmethod
    def add_sample(
        self, timestamp: float, sensor_name: str, data: Dict[str, Any]
    ) -> None:
        """Add a sensor sample at the given timestamp.

        Args:
            timestamp: The timestamp of the sample.
            sensor_name: The name of the sensor.
            data: The sensor data as a dictionary.

        Raises:
            DuplicateValueError: If data already exists at this timestamp for
                this sensor with different values.
            StaleTimestampError: If the timestamp is older than the last
                model request timestamp.
        """
        pass

    @abstractmethod
    def get_samples_up_to(
        self, timestamp: float
    ) -> List[Tuple[float, Dict[str, Dict[str, Any]]]]:
        """Get all samples up to and including the given timestamp.

        Removes the returned samples from the buffer and updates the
        last request timestamp for staleness checking.

        Args:
            timestamp: The cutoff timestamp (inclusive).

        Returns:
            A list of tuples (timestamp, sensor_data) sorted by timestamp,
            where sensor_data is a dict mapping sensor names to their data.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all buffered samples and reset state."""
        pass


class SequentialBucketsPolicy(TimePolicy):
    """Time policy that groups samples into sequential time buckets.

    Samples are organized by timestamp, with multiple sensors allowed at each
    timestamp. Enforces monotonicity per sensor and rejects stale timestamps
    after the first model request.
    """

    def __init__(self) -> None:
        self._buckets: Dict[float, Dict[str, Dict[str, Any]]] = {}
        self._last_request_timestamp: Optional[float] = None

    def add_sample(
        self, timestamp: float, sensor_name: str, data: Dict[str, Any]
    ) -> None:
        """Add a sensor sample at the given timestamp.

        Args:
            timestamp: The timestamp of the sample.
            sensor_name: The name of the sensor.
            data: The sensor data as a dictionary.

        Raises:
            DuplicateValueError: If data already exists at this timestamp for
                this sensor with different values.
            StaleTimestampError: If the timestamp is older than the last
                model request timestamp.
        """
        # Check staleness
        if (
            self._last_request_timestamp is not None
            and timestamp < self._last_request_timestamp
        ):
            raise StaleTimestampError(
                f"Timestamp {timestamp} is older than last request timestamp "
                f"{self._last_request_timestamp}"
            )

        # Check for duplicate values at same timestamp
        if timestamp in self._buckets and sensor_name in self._buckets[timestamp]:
            existing_data = self._buckets[timestamp][sensor_name]
            if existing_data != data:
                raise DuplicateValueError(
                    f"Conflicting data at timestamp {timestamp} for sensor "
                    f"'{sensor_name}': existing={existing_data}, new={data}"
                )
            # Same data, nothing to do
            return

        # Add to bucket
        if timestamp not in self._buckets:
            self._buckets[timestamp] = {}
        self._buckets[timestamp][sensor_name] = data

    def get_samples_up_to(
        self, timestamp: float
    ) -> List[Tuple[float, Dict[str, Dict[str, Any]]]]:
        """Get all samples up to and including the given timestamp.

        Removes the returned samples from the buffer and updates the
        last request timestamp for staleness checking.

        Args:
            timestamp: The cutoff timestamp (inclusive).

        Returns:
            A list of tuples (timestamp, sensor_data) sorted by timestamp,
            where sensor_data is a dict mapping sensor names to their data.
        """
        # Collect samples up to the given timestamp
        result: List[Tuple[float, Dict[str, Dict[str, Any]]]] = []
        timestamps_to_remove: List[float] = []

        for ts in sorted(self._buckets.keys()):
            if ts <= timestamp:
                result.append((ts, self._buckets[ts]))
                timestamps_to_remove.append(ts)
            else:
                break

        # Remove collected samples from buffer
        for ts in timestamps_to_remove:
            del self._buckets[ts]

        # Update last request timestamp
        self._last_request_timestamp = timestamp

        return result

    def clear(self) -> None:
        """Clear all buffered samples and reset state."""
        self._buckets.clear()
        self._last_request_timestamp = None


def create_time_policy(policy_name: str) -> TimePolicy:
    """Factory function to create a time policy by name.

    Args:
        policy_name: The name of the policy to create.
            Supported values: 'sequential_buckets'

    Returns:
        A TimePolicy instance.

    Raises:
        ValueError: If the policy name is not recognized.
    """
    policies = {
        "sequential_buckets": SequentialBucketsPolicy,
    }

    if policy_name not in policies:
        raise ValueError(
            f"Unknown time policy '{policy_name}'. "
            f"Supported policies: {list(policies.keys())}"
        )

    return policies[policy_name]()
