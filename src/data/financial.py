import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import Config

class FinancialLabeler:
    def __init__(self, config: Config):
        self.config = config
        self.attack_costs = self._load_costs()

    def _load_costs(self) -> pd.DataFrame:
        costs_path = Path(self.config.paths['data_financial']) / 'attack_costs.csv'
        if costs_path.exists():
            df = pd.read_csv(costs_path)
        else:
            base_costs = self.config.get('financial.base_costs', {})
            df = pd.DataFrame([
                {'attack_type': k, 'base_cost_usd': v}
                for k, v in base_costs.items()
            ])
        df['base_cost_usd'] = pd.to_numeric(df['base_cost_usd'], errors='coerce').fillna(0.0)
        df['base_cost_usd'] = df['base_cost_usd'].clip(lower=0.0)
        return df

    def get_base_cost(self, attack_type: str) -> float:
        cost_row = self.attack_costs[self.attack_costs['attack_type'] == attack_type]
        if not cost_row.empty:
            val = cost_row['base_cost_usd'].values[0]
            return float(val) if pd.notna(val) else float(self.attack_costs['base_cost_usd'].mean())
        else:
            return float(self.attack_costs['base_cost_usd'].mean())

    def calculate_intensity(self, df: pd.DataFrame) -> pd.Series:
        if 'Flow Packets/s' in df.columns and 'Flow Bytes/s' in df.columns:
            packets = pd.to_numeric(df['Flow Packets/s'], errors='coerce').fillna(0.0)
            bytes_ = pd.to_numeric(df['Flow Bytes/s'], errors='coerce').fillna(0.0)
            packets = packets.clip(lower=0.0)
            bytes_ = bytes_.clip(lower=0.0)
            max_packets = packets.quantile(0.99)
            max_bytes = bytes_.quantile(0.99)
            intensity = (
                (packets / (max_packets + 1e-8)) * 0.5 +
                (bytes_ / (max_bytes + 1e-8)) * 0.5
            )
            return intensity.clip(lower=0.5, upper=2.0)
        else:
            return pd.Series(1.0, index=df.index)

    def assign_detection_time(self, df: pd.DataFrame) -> pd.Series:
        label_col = self.config.data['target_column']
        complexity = {
            'BENIGN': 'fast',
            'PortScan': 'fast',
            'DoS Hulk': 'medium',
            'DDoS': 'medium',
            'DoS GoldenEye': 'medium',
            'DoS Slowloris': 'slow',
            'DoS Slowhttptest': 'slow',
            'FTP-Patator': 'medium',
            'SSH-Patator': 'medium',
            'Bot': 'slow',
            'Web Attack Brute Force': 'medium',
            'Web Attack XSS': 'slow',
            'Web Attack Sql Injection': 'slow',
            'Heartbleed': 'slow',
            'Infiltration': 'slow'
        }
        return df[label_col].map(complexity).fillna('medium')

    def assign_company_size(self, n_samples: int) -> np.ndarray:
        return np.full(n_samples, 'medium')

    def calculate_total_loss(self, base_cost: float, detection_time: str,
                             company_size: str, intensity: float) -> float:
        time_mult = self.config.get(f'financial.detection_time_multiplier.{detection_time}', 1.0)
        size_mult = self.config.get(f'financial.company_size_multiplier.{company_size}', 1.0)
        loss = base_cost * time_mult * size_mult * intensity
        return max(0.0, float(loss))

    def add_financial_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        label_col = self.config.data['target_column']
        df = df.copy()
        df['base_financial_loss'] = df[label_col].apply(self.get_base_cost)
        df['intensity_multiplier'] = self.calculate_intensity(df)
        df['detection_time'] = self.assign_detection_time(df)
        df['company_size'] = self.assign_company_size(len(df))
        df['total_financial_loss'] = df.apply(
            lambda row: self.calculate_total_loss(
                row['base_financial_loss'],
                row['detection_time'],
                row['company_size'],
                row['intensity_multiplier']
            ),
            axis=1
        )
        df['total_financial_loss'] = pd.to_numeric(df['total_financial_loss'], errors='coerce').fillna(0.0)
        df['total_financial_loss'] = df['total_financial_loss'].clip(lower=0.0, upper=100000.0)
        return df

    def compute_loss_by_attack_type(self, df: pd.DataFrame) -> dict:
        label_col = self.config.data['target_column']
        loss_by_label = df.groupby(label_col)['total_financial_loss'].mean().to_dict()
        return loss_by_label
