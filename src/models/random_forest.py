import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path
from typing import Dict, Optional
from src.interfaces.base import IModel
from src.utils.config import Config

class RandomForestModel(IModel):
    def __init__(self, config: Config, task: str = 'classification'):
        self.config = config
        self.task = task
        self.model = None
        self.params = config.get('models.random_forest.classification', {}).copy()

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        if isinstance(y_train, np.ndarray) and y_train.ndim > 1:
            y_train = y_train.ravel()
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        train_pred = self.model.predict(X_train)
        return {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_f1': f1_score(y_train, train_pred, average='weighted')
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, label_encoder=None) -> Dict:
        if isinstance(y_test, np.ndarray) and y_test.ndim > 1:
            y_test = y_test.ravel()
        y_pred = self.model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'predictions': y_pred
        }

    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> Dict:
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return {
            feature_names[idx] if idx < len(feature_names) else f"feature_{idx}": float(importances[idx])
            for idx in indices
        }

    def save(self):
        path = Path(self.config.paths['models']) / 'random_forest'
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / 'classification_model.pkl')
