import logging
from collections import Counter
from typing import Any

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from app.infrastructure.models.base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
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
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=-1),
        ]

        fit_kwargs: dict[str, Any] = {}

        if self.task == "classification":
            counts = Counter(y_train)
            total = len(y_train)
            n_classes = len(counts)
            sample_weights = np.array(
                [total / (n_classes * counts[y]) for y in y_train]
            )
            fit_kwargs["sample_weight"] = sample_weights
            logger.info(
                "LightGBM sample weights: min=%.2f, max=%.2f, ratio=%.1fx",
                sample_weights.min(),
                sample_weights.max(),
                sample_weights.max() / sample_weights.min(),
            )

        if eval_set:
            fit_kwargs["eval_set"] = eval_set

        self._model.fit(x_train_df, y_train, callbacks=callbacks, **fit_kwargs)
