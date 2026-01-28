from dataclasses import make_dataclass
from ..builder import BuildModel
from ..models.base import PredictionModel
from typing import Type


class ModelBuilder:

    # The type that will be built
    MODEL_CLS: Type[PredictionModel] = NotImplemented  # type: ignore

    def __init__(self, build_model: BuildModel) -> None:
        self.build_model = build_model

    def build(self) -> PredictionModel:
        raise NotImplementedError("Subclasses must implement the build method.")
    
    def _make_prediction_type(self):
        fields = []
        for name, parameter in self.build_model._predicted.items():
            fields.append((name, parameter.get_prediction_type()))
        return make_dataclass("PredictionType", fields)