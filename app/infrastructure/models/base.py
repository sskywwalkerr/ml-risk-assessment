import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from app.application.interfaces.model import EvalResult, IExplainableModel, TrainResult

logger = logging.getLogger(__name__)


class BaseModel(IExplainableModel):
    """Общая логика для всех моделей классификации и регрессии."""

    def __init__(self, task: str = "classification") -> None:
        self.task = task
        self._model: Any = None
        self._feature_names: list[str] = []

    @abstractmethod
    def _build(self) -> Any:
        """Создает и возвращает сконфигурированную модель."""
        ...

    def set_feature_names(self, feature_names: list[str]) -> None:
        """Сохраняет имена признаков до обучения."""
        self._feature_names = list(feature_names)

    def _to_frame(self, x: np.ndarray) -> pd.DataFrame:
        """Конвертирует numpy array в DataFrame с именами признаков."""
        if self._feature_names:
            return pd.DataFrame(x, columns=self._feature_names)
        return pd.DataFrame(x)

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
        """Базовый fit — переопределяется моделями с early stopping."""
        self._model.fit(self._to_frame(x_train), y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Возвращает предсказания модели."""
        return self._model.predict(self._to_frame(x))  # type: ignore[no-any-return]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Возвращает вероятности классов для ROC-кривой."""
        if not hasattr(self._model, "predict_proba"):
            raise NotImplementedError(
                f"{type(self._model).__name__} не поддерживает predict_proba"
            )
        return self._model.predict_proba(self._to_frame(x))  # type: ignore[no-any-return]

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> EvalResult:
        """Вычисляет метрики качества на тестовой выборке."""
        y_pred = self.predict(x_test)
        if self.task == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            }
        else:
            metrics = {}
        return EvalResult(predictions=y_pred, metrics=metrics)

    def get_feature_importance(
        self, feature_names: list[str], top_n: int = 15
    ) -> dict[str, float]:
        """Возвращает топ-N признаков по важности."""
        if not hasattr(self._model, "feature_importances_"):
            return {}
        imp: np.ndarray = self._model.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        return {
            (feature_names[i] if i < len(feature_names) else f"feature_{i}"): float(
                imp[i]
            )
            for i in idx
        }

    def save(self, path: str) -> None:
        """Сохраняет модель и имена признаков на диск."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, p / f"{self.__class__.__name__}_{self.task}.pkl")
        joblib.dump(
            self._feature_names,
            p / f"{self.__class__.__name__}_{self.task}_features.pkl",
        )
        logger.info("[%s] Saved to %s", self.__class__.__name__, path)
