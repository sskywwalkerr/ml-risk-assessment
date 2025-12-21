import pandas as pd
import numpy as np
from src.interfaces.base import IFeatureEngineer
from src.utils.config import Config

class FeatureEngineer(IFeatureEngineer):
    def __init__(self, config: Config):
        self.config = config

    def create_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('features.create_ratios', True):
            return df
        df = df.copy()
        if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
            fwd = pd.to_numeric(df['Total Fwd Packets'], errors='coerce').fillna(0.0)
            bwd = pd.to_numeric(df['Total Backward Packets'], errors='coerce').fillna(0.0)
            df['Fwd_Bwd_Packet_Ratio'] = fwd / (bwd + 1e-8)
        if 'Fwd Packet Length Mean' in df.columns and 'Bwd Packet Length Mean' in df.columns:
            fwd_len = pd.to_numeric(df['Fwd Packet Length Mean'], errors='coerce').fillna(0.0)
            bwd_len = pd.to_numeric(df['Bwd Packet Length Mean'], errors='coerce').fillna(0.0)
            df['Fwd_Bwd_Length_Ratio'] = fwd_len / (bwd_len + 1e-8)
        if 'Flow Packets/s' in df.columns and 'Flow Bytes/s' in df.columns:
            packets = pd.to_numeric(df['Flow Packets/s'], errors='coerce').fillna(0.0)
            bytes_ = pd.to_numeric(df['Flow Bytes/s'], errors='coerce').fillna(0.0)
            df['Attack_Intensity'] = packets * bytes_
        return df

    def create_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('features.create_statistical', True):
            return df
        df = df.copy()
        if 'Flow IAT Mean' in df.columns and 'Flow IAT Std' in df.columns:
            mean = pd.to_numeric(df['Flow IAT Mean'], errors='coerce').fillna(0.0)
            std = pd.to_numeric(df['Flow IAT Std'], errors='coerce').fillna(0.0)
            df['IAT_Coefficient_Variation'] = std / (mean + 1e-8)
        if 'Fwd IAT Total' in df.columns and 'Bwd IAT Total' in df.columns:
            fwd_iat = pd.to_numeric(df['Fwd IAT Total'], errors='coerce').fillna(0.0)
            bwd_iat = pd.to_numeric(df['Bwd IAT Total'], errors='coerce').fillna(0.0)
            df['Traffic_Asymmetry'] = abs(fwd_iat - bwd_iat)
        if 'Flow Bytes/s' in df.columns and 'Flow Packets/s' in df.columns:
            bytes_ = pd.to_numeric(df['Flow Bytes/s'], errors='coerce').fillna(0.0)
            packets = pd.to_numeric(df['Flow Packets/s'], errors='coerce').fillna(0.0)
            df['Flow_Density'] = bytes_ / (packets + 1e-8)
        return df

    def create_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('features.create_temporal', True):
            return df
        df = df.copy()
        if 'Flow Duration' in df.columns:
            duration = pd.to_numeric(df['Flow Duration'], errors='coerce').fillna(0.0)
            duration = np.clip(duration, 0, None)
            df['Flow_Duration_Log'] = np.log1p(duration)
        return df

    def create_anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('features.create_anomaly_scores', True):
            return df
        df = df.copy()
        cols = ['Flow Bytes/s', 'Flow Packets/s', 'Flow Duration']
        for col in cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                mean = series.mean()
                std = series.std()
                if pd.isna(mean) or pd.isna(std) or std == 0:
                    df[f'{col}_Zscore'] = 0.0
                else:
                    df[f'{col}_Zscore'] = abs((series - mean) / (std + 1e-10))
        return df

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.create_ratios(df)
        df = self.create_statistical(df)
        df = self.create_temporal(df)
        df = self.create_anomaly_scores(df)
        return df
