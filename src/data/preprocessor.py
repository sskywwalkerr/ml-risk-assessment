import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from src.interfaces.base import IPreprocessor
from src.utils.config import Config

class Preprocessor(IPreprocessor):
    def __init__(self, config: Config):
        self.config = config
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.target_column = config.data['target_column']
        self.feature_names = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        threshold = len(numeric_cols) * 0.5
        df_clean = df_clean.dropna(thresh=int(threshold))
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df_clean[col] = df_clean[col].fillna(median_val)
        return df_clean.drop_duplicates()

    def preprocess(self, df: pd.DataFrame) -> dict:
        df_clean = self.clean_data(df)
        y = df_clean[self.target_column].copy()
        X = df_clean.drop(columns=[self.target_column])
        financial_cols = ['base_financial_loss', 'intensity_multiplier',
                         'detection_time', 'company_size', 'total_financial_loss']
        X = X.drop(columns=[c for c in financial_cols if c in X.columns])
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        self.feature_names = numeric_cols
        
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
        )
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'feature_names': numeric_cols, 'label_encoder': self.label_encoder,
            'n_classes': len(self.label_encoder.classes_)
        }

    def save(self, path: str = "models/metadata"):
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, save_path / 'scaler.pkl')
        joblib.dump(self.label_encoder, save_path / 'label_encoder.pkl')
        
        if self.feature_names is not None:
            joblib.dump(self.feature_names, save_path / 'feature_names.pkl')
            print(f"  Сохранено {len(self.feature_names)} feature names")
        else:
            print(f"  WARNING: feature_names не установлены, пропуск сохранения")

