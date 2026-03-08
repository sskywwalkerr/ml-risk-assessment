from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (  # type: ignore
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from app.application.interfaces.model import IExplainableModel
from app.application.interfaces.model_repository import IModelRepository


@dataclass(frozen=True, slots=True)
class TrainRegressorRequest:
    """Параметры для обучения регрессора финансовых потерь."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    model_name: str


@dataclass(frozen=True, slots=True)
class TrainRegressorResponse:
    """Результат обучения регрессора."""

    model_name: str
    test_metrics: dict  # mae, rmse, r2, mape в реальных USD
    feature_importance: dict
    y_pred: np.ndarray
    y_test: np.ndarray


class TrainRegressorInteractor:
    """Обучает регрессор для предсказания финансовых потерь."""

    def __init__(
        self,
        model: IExplainableModel,
        repository: IModelRepository,
    ) -> None:
        self._model = model
        self._repository = repository

    def __call__(self, request: TrainRegressorRequest) -> TrainRegressorResponse:
        y_train_log = np.log1p(request.y_train)
        y_val_log = np.log1p(request.y_val)

        self._model.train(
            request.x_train,
            y_train_log,
            request.x_val,
            y_val_log,
        )
        result = self._model.evaluate(request.x_test, np.log1p(request.y_test))

        y_pred = np.expm1(result.predictions)
        y_true = request.y_test
        mask = y_true > 0

        metrics = {
            "mae": float(
                mean_absolute_error(y_true, y_pred)
            ),  # Средняя абсолютная ошибка
            "rmse": float(
                np.sqrt(mean_squared_error(y_true, y_pred))
            ),  # Корень из среднеквадратичной ошибки
            "r2": float(r2_score(y_true, y_pred)),  # Коэффициент детерминации
            "mape": (
                float(
                    np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                )
                if mask.sum() > 0
                else 0.0
            ),  # Средняя относительная ошибка
        }

        importance = self._model.get_feature_importance(request.feature_names)
        self._repository.save(self._model, request.model_name)

        return TrainRegressorResponse(
            model_name=request.model_name,
            test_metrics=metrics,
            feature_importance=importance,
            y_pred=y_pred,
            y_test=y_true,
        )
