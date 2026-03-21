from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from app.infrastructure.models.base import BaseModel


class LightGBMModel(BaseModel):
    """Обучение LightGBM."""

    _PARAMS: dict[str, dict[str, Any]] = {
        "classification": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        },
        "regression": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        },
    }

    def _build(self) -> object:
        params = self._PARAMS[self.task]
        return (
            LGBMClassifier(**params)
            if self.task == "classification"
            else LGBMRegressor(**params)
        )

    def _fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        """Использует eval_set(данные, ответы) для проверки."""
        eval_set = [(x_val, y_val)] if x_val is not None else None
        self._model.fit(
            x_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
