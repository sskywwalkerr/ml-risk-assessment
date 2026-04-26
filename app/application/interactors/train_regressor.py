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
    test_metrics: dict  # mae, rmse, r2, mape
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
        self._model.set_feature_names(request.feature_names)

        self._model.train(
            request.x_train,
            request.y_train,
            request.x_val,
            request.y_val,
        )
        result = self._model.evaluate(request.x_test, request.y_test)

        y_pred = result.predictions
        y_true = request.y_test
        mask = y_true > 0

        metrics = {
            "mae": float(
                mean_absolute_error(y_true, y_pred)
            ),  # средняя абсолютная ошибка
            "rmse": float(
                np.sqrt(mean_squared_error(y_true, y_pred))
            ),  # корень из среднеквадратичной
            "r2": float(r2_score(y_true, y_pred)),  # коэффициент детерминации
            # MAPE считаем только по атакам (ущерб > 0) — иначе деление на 0
            "mape": (
                float(
                    np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                )
                if mask.sum() > 0
                else 0.0
            ),  # средняя относительная ошибка (только по атакам, где ущерб > 0)
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
