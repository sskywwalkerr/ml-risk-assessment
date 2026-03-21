from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

from app.application.interfaces.preprocessor import DataSplits, IPreprocessor

# Колонки, которые не являются признаками
_NON_FEATURE_COLS = {
    "label",
    "base_financial_loss",
    "intensity_multiplier",
    "direct_loss",
    "indirect_loss",
    "reputation_loss",
    "regulatory_fine",
    "total_financial_loss",
    "risk_level",
}


class RobustPreprocessor(IPreprocessor):
    """Очищает данные, масштабирует RobustScaler и разбивает на выборки."""

    def __init__(self) -> None:
        self._scaler: RobustScaler = RobustScaler()
        self._encoder: LabelEncoder = LabelEncoder()
        self._feature_names: list[str] = []

    def preprocess(self, df: pd.DataFrame) -> DataSplits:
        df = self._clean(df)

        y = self._encoder.fit_transform(df["label"])

        drop = [c for c in df.columns if c in _NON_FEATURE_COLS]

        x = df.drop(columns=drop).select_dtypes(include=[np.number])
        self._feature_names = x.columns.tolist()

        x_scaled = self._scaler.fit_transform(x)

        # 70% train / 10% val / 20% test
        x_tmp, x_test, y_tmp, y_test = train_test_split(
            x_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_tmp, y_tmp, test_size=0.125, random_state=42, stratify=y_tmp
        )

        return DataSplits(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=tuple(self._feature_names),
            label_encoder=self._encoder,
            n_classes=len(self._encoder.classes_),
        )

    def save(self, path: str) -> None:
        """Сохранение для инференса."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, p / "scaler.pkl")
        joblib.dump(self._encoder, p / "label_encoder.pkl")
        joblib.dump(self._feature_names, p / "feature_names.pkl")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаляет inf, заполняет NaN медианой, убирает дубликаты."""
        df = df.copy()
        num = df.select_dtypes(include=[np.number]).columns

        df[num] = df[num].replace([np.inf, -np.inf], np.nan)

        # >50% это пустые
        df = df.dropna(thresh=int(len(num) * 0.5))

        for col in num:
            if df[col].isnull().any():
                median = df[col].median()
                df[col] = df[col].fillna(median if not np.isnan(median) else 0.0)
        return df.drop_duplicates()
