
from typing import Type
from noisytracking.models.base import AnyDataclass, PredictionModel
from noisytracking.time_policy import TimePolicy


class PyroModel(PredictionModel):
    
    def __init__(self, sample_provider: TimePolicy, prediction_type: Type[AnyDataclass]) -> None:
        self.sample_provider = sample_provider
        self.prediction_type = prediction_type
        raise NotImplementedError("PyroModel initialization is not implemented yet.")
        super().__init__()

    def predict(self, timestamp: float) -> AnyDataclass:
        raise NotImplementedError("Prediction not implemented.")
        return self.prediction_type(...)
    
    def update(self, timestamp: float) -> None:
        raise NotImplementedError("Update not implemented.")
