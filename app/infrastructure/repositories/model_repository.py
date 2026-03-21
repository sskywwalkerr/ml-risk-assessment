from pathlib import Path

from app.application.interfaces.model import IModel
from app.application.interfaces.model_repository import IModelRepository
from app.infrastructure.exceptions import ModelLoadError


class FileModelRepository(IModelRepository):
    """Хранилище моделей."""

    def __init__(self, base_path: str = "models") -> None:
        self._path = Path(base_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._registry: dict[str, IModel] = {}

    def save(self, model: IModel, name: str) -> None:
        """Сохраняет модель и регистрирует в памяти."""
        model_path = self._path / name
        model.save(str(model_path))
        self._registry[name] = model

    def load(self, name: str) -> IModel:
        """Загружает модель из памяти по имени."""
        if name in self._registry:
            return self._registry[name]
        raise ModelLoadError(name)

    def list(self) -> list[str]:
        """Возвращает список всех сохраненных моделей."""
        return list(self._registry.keys())
