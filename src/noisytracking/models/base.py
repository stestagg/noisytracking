
from typing import Any, Protocol, Type
from ..time_policy import TimePolicy


class AnyDataclass(Protocol):
    __dataclass_fields__: dict[str, Any]


class PredictionModel:
    sample_provider: TimePolicy

    def predict(self, timestamp: float) -> AnyDataclass:
        raise NotImplementedError("Prediction not implemented.")
    
    def update(self, timestamp: float) -> None:
        raise NotImplementedError("Update not implemented.")
    
    def update_from_sensor(self, timestamp: float, **sensor_data) -> None:
        self.sample_provider.add_sample(timestamp, **sensor_data)