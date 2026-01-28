
from dataclasses import make_dataclass
from ..builder import BuildModel
from .base import ModelBuilder
from ..models.pyro_model import PyroModel


class PyroBuilder(ModelBuilder):
    
    MODEL_CLS = PyroModel

    def build(self) -> PyroModel:
        raise NotImplementedError("PyroModel building is not implemented yet.")
