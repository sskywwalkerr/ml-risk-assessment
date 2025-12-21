from pathlib import Path
from typing import Dict

from src.interfaces.base import IModelRepository, IModel


class ModelRepository(IModelRepository):
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, IModel] = {}

    def save_model(self, model: IModel, name: str) -> None:
        model_path = self.base_path / name
        model_path.mkdir(parents=True, exist_ok=True)
        self._models[name] = model

    def load_model(self, name: str) -> IModel:
        return self._models.get(name)

    def list_models(self) -> list:
        return list(self._models.keys())
