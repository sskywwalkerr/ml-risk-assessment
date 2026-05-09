import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.application.interfaces.preprocessor import DataSplits, IPreprocessor

logger = logging.getLogger(__name__)

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

_ZSCORE_COLS = ("rate", "header_length", "flow_duration", "iat")

# Порог клиппинга после масштабирования — убирает экстремальные выбросы
# не ломая информацию о редких классах
_CLIP_VALUE = 10.0


class RobustPreprocessor(IPreprocessor):
    """
    Предобработка с StandardScaler + клиппинг выбросов.

    Почему не RobustScaler:
      RobustScaler использует медиану и IQR. При дисбалансе 99:1
      медиана и IQR определяются доминирующим классом (DDoS).
      Признаки редких классов (BruteForce rst_count=9613) после
      масштабирования получают значения ~961,300 — модель их не видела
      при обучении и сваливается в DDoS по умолчанию.

    StandardScaler использует mean/std по всей выборке, что даёт
    более равномерное масштабирование. Клиппинг [-10, 10] убирает
    оставшиеся выбросы без потери информации о классе.
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler = StandardScaler()
        self._encoder: LabelEncoder = LabelEncoder()
        self._feature_names: list[str] = []
        self._zscore_stats: dict[str, dict[str, float]] = {}
        self.clean_df: pd.DataFrame | None = None
        self.train_idx: np.ndarray = np.array([])
        self.val_idx: np.ndarray = np.array([])
        self.test_idx: np.ndarray = np.array([])

    def preprocess(self, df: pd.DataFrame) -> DataSplits:
        logger.info(
            "Запуск предобработки: %d строк, %d колонок", len(df), len(df.columns)
        )
        df = self._clean(df)
        df = df.reset_index(drop=True)
        self.clean_df = df
        logger.info("После очистки: %d строк", len(df))

        y = self._encoder.fit_transform(df["label"])
        logger.info(
            "Классов: %d -> %s",
            len(self._encoder.classes_),
            list(self._encoder.classes_),
        )

        # Сохраняем zscore статистики до скейлинга
        for col in _ZSCORE_COLS:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                mean_val = float(s.mean())
                std_val = float(s.std())
                self._zscore_stats[col] = {
                    "mean": mean_val,
                    "std": std_val if std_val > 0 else 1.0,
                }
                logger.info(
                    "zscore_stats[%s]: mean=%.4f std=%.4f", col, mean_val, std_val
                )

        drop = [c for c in df.columns if c in _NON_FEATURE_COLS]
        x = df.drop(columns=drop).select_dtypes(include=[np.number])
        self._feature_names = x.columns.tolist()
        logger.info("Признаков: %d", len(self._feature_names))

        # StandardScaler: (x - mean) / std
        x_scaled = self._scaler.fit_transform(x)

        # Клиппинг: убираем экстремальные выбросы после масштабирования
        # [-10, 10] покрывает 99.9%+ нормального распределения
        x_scaled = np.clip(x_scaled, -_CLIP_VALUE, _CLIP_VALUE)
        logger.info("Клиппинг выбросов: [%.1f, %.1f]", -_CLIP_VALUE, _CLIP_VALUE)

        idx = np.arange(len(x_scaled))

        idx_tmp, idx_test, x_tmp, x_test, y_tmp, y_test = train_test_split(
            idx, x_scaled, y,
            test_size=0.2, random_state=42, stratify=y,
        )
        idx_train, idx_val, x_train, x_val, y_train, y_val = train_test_split(
            idx_tmp, x_tmp, y_tmp,
            test_size=0.125, random_state=42, stratify=y_tmp,
        )

        self.train_idx = idx_train
        self.val_idx = idx_val
        self.test_idx = idx_test

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
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._scaler, p / "scaler.pkl")
        joblib.dump(self._encoder, p / "label_encoder.pkl")
        joblib.dump(self._feature_names, p / "feature_names.pkl")
        joblib.dump(self._zscore_stats, p / "zscore_stats.pkl")
        # Сохраняем clip_value чтобы инференс знал о нём
        joblib.dump(_CLIP_VALUE, p / "clip_value.pkl")
        logger.info("Сохранены zscore_stats для %d признаков", len(self._zscore_stats))

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        num = df.select_dtypes(include=[np.number]).columns
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(thresh=int(len(num) * 0.5))
        for col in num:
            if df[col].isnull().any():
                median = df[col].median()
                df[col] = df[col].fillna(median if not np.isnan(median) else 0.0)
        return df