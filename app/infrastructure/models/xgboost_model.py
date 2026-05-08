from collections import Counter
from typing import Any

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from app.infrastructure.models.base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(
        self, task: str = "classification", params: dict[str, Any] | None = None
    ) -> None:
        super().__init__(task)
        self._params: dict[str, Any] = (params or {}).copy()
        self._params.setdefault("device", "cuda")
        self._params.setdefault("early_stopping_rounds", 50)

    def _build(self) -> Any:
        if self.task == "classification":
            return XGBClassifier(**self._params)
        return XGBRegressor(**self._params)

    def _fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        x_train_df = self._to_frame(x_train)
        fit_kwargs: dict[str, Any] = {"verbose": False}

        if self.task == "classification":
            counts = Counter(y_train)
            total = len(y_train)
            n_classes = len(counts)

            sample_weights = np.array(
                [total / (n_classes * counts[y]) for y in y_train]
            )
            fit_kwargs["sample_weight"] = sample_weights

            min_w = sample_weights.min()
            max_w = sample_weights.max()
            import logging

            logging.getLogger(__name__).info(
                "Sample weights: min=%.2f, max=%.2f, ratio=%.1fx",
                min_w,
                max_w,
                max_w / min_w,
            )

        if x_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(self._to_frame(x_val), y_val)]

        self._model.fit(x_train_df, y_train, **fit_kwargs)
