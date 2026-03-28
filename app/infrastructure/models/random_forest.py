from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.infrastructure.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest -базовая линия для сравнения моделей."""

    def __init__(
        self, task: str = "classification", params: dict[str, Any] | None = None
    ) -> None:
        super().__init__(task)
        self._params: dict[str, Any] = params or {}

    def _build(self) -> Any:
        return (
            RandomForestClassifier(**self._params)
            if self.task == "classification"
            else RandomForestRegressor(**self._params)
        )
