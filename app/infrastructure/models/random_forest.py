from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.infrastructure.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest -базовая линия для сравнения моделей."""

    _PARAMS = {
        "classification": {
            "n_estimators": 200,  # деревья
            "max_depth": 20,  # предел глубины дерева
            "min_samples_split": 5,  # минимальное количество объектов, необходимое для того, чтобы узел дерева мог разделиться на два дочерних
            "n_jobs": -1,  # Использование ядер процессора
            "random_state": 42,  # Фиксация случайности
            "class_weight": "balanced",  # баланс для редких классов
        },
        "regression": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "n_jobs": -1,
            "random_state": 42,
        },
    }

    def _build(self) -> object:
        params = self._PARAMS[self.task]
        return (
            RandomForestClassifier(**params)
            if self.task == "classification"
            else RandomForestRegressor(**params)
        )
