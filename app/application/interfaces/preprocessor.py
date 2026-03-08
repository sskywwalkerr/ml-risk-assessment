from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # type: ignore


@dataclass(frozen=True, slots=True)
class DataSplits:
    """Результат предобработки."""

    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    label_encoder: LabelEncoder
    n_classes: int


class IPreprocessor(ABC):
    """Интерфейс для очистки, масштабирования и разбивки данных."""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> DataSplits:
        """Очищает данные, масштабирует и делит на train/val/test."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Сохраняет scaler и label_encoder на диск."""
        ...
