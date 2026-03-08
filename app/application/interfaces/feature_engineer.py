from abc import ABC, abstractmethod

import pandas as pd


class IFeatureEngineer(ABC):
    """Интерфейс для создания новых признаков из сырого трафика."""

    @abstractmethod
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создаёт производные признаки и возвращает расширенный DataFrame."""
        ...
