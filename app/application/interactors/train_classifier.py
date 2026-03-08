from dataclasses import dataclass

import numpy as np

from app.application.interfaces.model import IExplainableModel
from app.application.interfaces.model_repository import IModelRepository


@dataclass(frozen=True, slots=True)
class TrainClassifierRequest:
    """Параметры для обучения классификатора атак."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    model_name: str


@dataclass(frozen=True, slots=True)
class TrainClassifierResponse:
    """Результат обучения классификатора."""

    model_name: str
    test_metrics: dict  # accuracy, f1_score
    feature_importance: dict  # топ-15 признаков
    y_pred: np.ndarray
    y_test: np.ndarray


class TrainClassifierInteractor:
    """Обучает классификатор атак, оценивает и сохраняет модель."""

    def __init__(
        self,
        model: IExplainableModel,
        repository: IModelRepository,
    ) -> None:
        self._model = model
        self._repository = repository

    def __call__(self, request: TrainClassifierRequest) -> TrainClassifierResponse:
        self._model.train(
            request.x_train,
            request.y_train,
            request.x_val,
            request.y_val,
        )
        result = self._model.evaluate(request.x_test, request.y_test)
        importance = self._model.get_feature_importance(request.feature_names)

        self._repository.save(self._model, request.model_name)

        return TrainClassifierResponse(
            model_name=request.model_name,
            test_metrics=result.metrics,
            feature_importance=importance,
            y_pred=result.predictions,
            y_test=request.y_test,
        )
