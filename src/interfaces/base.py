from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import pandas as pd


class IDataLoader(ABC):
    @abstractmethod
    def load(self, use_sample: bool = False, sample_size: Optional[int] = None) -> pd.DataFrame:
        ...


class IPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> Dict:
        ...

    @abstractmethod
    def save(self, path: str = None) -> None:
        ...


class IFeatureEngineer(ABC):
    @abstractmethod
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class IModel(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        ...

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, label_encoder=None) -> Dict:
        ...

    @abstractmethod
    def save(self) -> None:
        ...

    @abstractmethod
    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> Dict:
        ...


class IModelRepository(ABC):
    @abstractmethod
    def save_model(self, model: IModel, name: str) -> None:
        ...

    @abstractmethod
    def load_model(self, name: str) -> IModel:
        ...

    @abstractmethod
    def list_models(self) -> list:
        ...
