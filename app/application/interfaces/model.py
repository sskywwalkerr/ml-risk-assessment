from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Результат обучения модели."""

    train_loss: float | None = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Результат оценки модели на тестовой выборке."""

    predictions: np.ndarray
    metrics: dict


class IModel(ABC):
    """Базовый интерфейс для всех ML-моделей."""

    @abstractmethod
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TrainResult:
        """Обучает модель на тренировочных данных."""
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Возвращает предсказания для входных данных."""
        ...

    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> EvalResult:
        """Вычисляет метрики на тестовой выборке."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Сохраняет модель на диск."""
        ...


class IExplainableModel(IModel, ABC):
    """Интерфейс для моделей, поддерживающих объяснимость и вероятности."""

    @abstractmethod
    def get_feature_importance(
        self,
        feature_names: list[str],
        top_n: int = 15,
    ) -> dict[str, float]:
        """Возвращает топ-N наиболее важных признаков."""
        ...

    @abstractmethod
    def set_feature_names(self, feature_names: list[str]) -> None:
        """Сохраняет имена признаков до обучения."""
        ...

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Возвращает вероятности классов (для ROC-кривой)."""
        ...
