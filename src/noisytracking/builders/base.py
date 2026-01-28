from ..builder import BuildModel
from ..models.base import PredictionModel
from typing import Type

class ModelBuilder:

    # A subclass of PredictionModel
    MODEL_CLS: Type[PredictionModel] = NotImplemented  # type: ignore

    def build(self, build_model: BuildModel) -> PredictionModel:
        raise NotImplementedError("Subclasses must implement build method.")