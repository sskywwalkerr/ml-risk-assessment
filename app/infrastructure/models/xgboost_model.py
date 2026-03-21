import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from app.infrastructure.models.base import BaseModel


class XGBoostModel(BaseModel):
    """Обучение XGBoost."""

    _PARAMS = {
        "classification": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,  # Скорость обучения (шаг градиента), меньше лучше результат
            "subsample": 0.8,  # Доля данных для обучения одного дерева
            "colsample_bytree": 0.8,  # Доля признаков (колонок) для одного дерева, лучшие признаки только из 80%
            "eval_metric": "mlogloss",  # Метрика качества, для более 2х классов
            "n_jobs": -1,  # Параллелизм
            "random_state": 42,
            "verbose": 0,  # Отчеты
        },
        "regression": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "rmse",  # Среднеквадратичная ошибка
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": 0,
        },
    }

    def _build(self) -> object:
        params = self._PARAMS[self.task]
        return (
            XGBClassifier(**params)
            if self.task == "classification"
            else XGBRegressor(**params)
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
