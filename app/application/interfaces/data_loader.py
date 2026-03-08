from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class LoadDataRequest:
    """Параметры загрузки датасета."""

    sample_size: int | None = None


class IDataLoader(ABC):
    """Интерфейс для загрузки датасета сетевого трафика."""

    @abstractmethod
    def load(self, request: LoadDataRequest) -> pd.DataFrame:
        """Загружает CSV файлы и возвращает сырой DataFrame."""
        ...
