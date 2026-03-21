import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from app.application.interfaces.model import EvalResult, IExplainableModel, TrainResult

logger = logging.getLogger(__name__)


class BaseModel(IExplainableModel):
    """Общая логика для всех моделей классификации и регрессии."""

    def __init__(self, task: str = "classification") -> None:
        self.task = task
        self._model: Any = None

    @abstractmethod
    def _build(self) -> object:
        """Создает и возвращает сконфигурированную модель."""
        ...

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TrainResult:
        """Строит и обучает модель."""
        self._model = self._build()
        self._fit(x_train, y_train, x_val, y_val)
        return TrainResult()

    def _fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        """Базовый fit - переопределяется моделями с early stopping."""
        self._model.fit(x_train, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Возвращает предсказания модели."""
        return self._model.predict(x)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> EvalResult:
        """Вычисляет метрики качества."""
        y_pred = self.predict(x_test)
        if self.task == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            }
        else:
            # Регрессия - метрики считает interactor после обратного log-преобразования
            metrics = {}
        return EvalResult(predictions=y_pred, metrics=metrics)

    def get_feature_importance(
        self, feature_names: list[str], top_n: int = 15
    ) -> dict[str, float]:
        """Возвращает топ-N признаков по важности."""
        if not hasattr(
            self._model, "feature_importances_"
        ):  # Проверка на признаки для RandomForest и XGBoost
            return {}
        imp = self._model.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        return {
            (feature_names[i] if i < len(feature_names) else f"feature_{i}"): float(
                imp[i]
            )
            for i in idx
        }

    def save(self, path: str) -> None:
        """Сохранение модели."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, p / f"{self.__class__.__name__}_{self.task}.pkl")
        logger.info(f"[{self.__class__.__name__}] Saved to {path}")
