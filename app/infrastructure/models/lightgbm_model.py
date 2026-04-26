from typing import Any

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from app.infrastructure.models.base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM быстрый градиентный бустинг."""

    def __init__(
        self, task: str = "classification", params: dict[str, Any] | None = None
    ) -> None:
        super().__init__(task)
        self._params: dict[str, Any] = params or {}

    def _build(self) -> Any:
        return (
            LGBMClassifier(**self._params)
            if self.task == "classification"
            else LGBMRegressor(**self._params)
        )

    def _fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        x_train_df = self._to_frame(x_train)
        eval_set = [(self._to_frame(x_val), y_val)] if x_val is not None else None
        callbacks = [lgb.log_evaluation(period=-1)]
        self._model.fit(x_train_df, y_train, eval_set=eval_set, callbacks=callbacks)
