from abc import ABC, abstractmethod

from app.application.interfaces.model import IModel


class IModelRepository(ABC):
    """Интерфейс для сохранения и загрузки обученных моделей."""

    @abstractmethod
    def save(self, model: IModel, name: str) -> None:
        """Сохраняет модель под заданным именем."""
        ...

    @abstractmethod
    def load(self, name: str) -> IModel:
        """Загружает модель по имени."""
        ...

    @abstractmethod
    def list(self) -> list[str]:
        """Возвращает список сохранённых моделей."""
        ...
