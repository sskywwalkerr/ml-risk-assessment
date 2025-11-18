import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config import Config
from src.data.loader import DataLoader
from src.features.engineering import FeatureEngineer

logger = get_logger(__name__)


class FinancialLabeler:
    """Класс для создания финансовых меток"""

    def __init__(self, config: Config):
        self.config = config
        self.attack_costs = self._load_costs()

    def _load_costs(self) -> pd.DataFrame:
        """Загрузка базовой стоимости атак"""

        costs_path = Path(self.config.paths['data_financial']) / 'attack_costs.csv'

        if costs_path.exists():
            logger.info(f"Загрузка стоимости атак: {costs_path}")
            return pd.read_csv(costs_path)
        else:
            logger.info(f"Использование стоимости из конфига")
            base_costs = self.config.get('financial.base_costs', {})
            return pd.DataFrame([
                {'attack_type': k, 'base_cost_usd': v}
                for k, v in base_costs.items()
            ])

    def get_base_cost(self, attack_type: str) -> float:
        """Получить базовую стоимость атаки"""

        cost_row = self.attack_costs[
            self.attack_costs['attack_type'] == attack_type
            ]

        if not cost_row.empty:
            return float(cost_row['base_cost_usd'].values[0])
        else:
            return float(self.attack_costs['base_cost_usd'].mean())

    def calculate_intensity(self, df: pd.DataFrame) -> pd.Series:
        """
        Расчет интенсивности атаки (множитель)

        Args:
            df: DataFrame с признаками

        Returns:
            Series с множителями интенсивности
        """
        if 'Flow Packets/s' in df.columns and 'Flow Bytes/s' in df.columns:
            # Нормализация
            max_packets = df['Flow Packets/s'].quantile(0.99)
            max_bytes = df['Flow Bytes/s'].quantile(0.99)

            intensity = (
                    (df['Flow Packets/s'] / (max_packets + 1)) * 0.5 +
                    (df['Flow Bytes/s'] / (max_bytes + 1)) * 0.5
            )

            # Ограничиваем диапазон
            return intensity.clip(0.5, 2.0)
        else:
            return pd.Series(1.0, index=df.index)

    def assign_detection_time(self, df: pd.DataFrame) -> pd.Series:
        """
        Симуляция времени обнаружения атаки

        Args:
            df: DataFrame

        Returns:
            Series с временем обнаружения (fast/medium/slow)
        """
        label_col = self.config.data['target_column']

        # Сложность обнаружения по типу атаки
        complexity = {
            'BENIGN': 'fast',
            'PortScan': 'fast',
            'DoS Hulk': 'medium',
            'DDoS': 'medium',
            'DoS GoldenEye': 'medium',
            'DoS Slowloris': 'slow',
            'DoS slowloris': 'slow',
            'DoS Slowhttptest': 'slow',
            'FTP-Patator': 'medium',
            'SSH-Patator': 'medium',
            'Bot': 'slow',
            'Web Attack - Brute Force': 'medium',
            'Web Attack - XSS': 'slow',
            'Web Attack - Sql Injection': 'slow',
            'Heartbleed': 'slow',
            'Infiltration': 'slow'
        }

        return df[label_col].map(complexity).fillna('medium')

    def assign_company_size(self, n_samples: int) -> np.ndarray:
        """
        Случайное распределение размера компании

        Args:
            n_samples: количество образцов

        Returns:
            Array с размерами компаний
        """
        np.random.seed(42)
        return np.random.choice(
            ['small', 'medium', 'large'],
            size=n_samples,
            p=[0.3, 0.5, 0.2]
        )

    def calculate_total_loss(
            self,
            base_cost: float,
            detection_time: str,
            company_size: str,
            intensity: float
    ) -> float:
        """
        Расчет общей финансовой потери

        Args:
            base_cost: базовая стоимость
            detection_time: время обнаружения
            company_size: размер компании
            intensity: интенсивность атаки

        Returns:
            Общая стоимость в USD
        """
        # Множители
        time_mult = self.config.get(
            f'financial.detection_time_multiplier.{detection_time}',
            1.0
        )
        size_mult = self.config.get(
            f'financial.company_size_multiplier.{company_size}',
            1.0
        )

        return base_cost * time_mult * size_mult * intensity

    def add_financial_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление всех финансовых меток

        Args:
            df: DataFrame с данными

        Returns:
            DataFrame с финансовыми метками
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"💰 СОЗДАНИЕ ФИНАНСОВЫХ МЕТОК")
        logger.info(f"{'=' * 70}\n")

        label_col = self.config.data['target_column']

        # Базовая стоимость
        logger.info("  • Расчет базовой стоимости...")
        df['base_financial_loss'] = df[label_col].apply(self.get_base_cost)

        # Интенсивность
        logger.info("  • Расчет интенсивности атаки...")
        df['intensity_multiplier'] = self.calculate_intensity(df)

        # Время обнаружения
        logger.info("  • Определение времени обнаружения...")
        df['detection_time'] = self.assign_detection_time(df)

        # Размер компании
        logger.info("  • Распределение размеров компаний...")
        df['company_size'] = self.assign_company_size(len(df))

        # Общая стоимость
        logger.info("  • Расчет общих финансовых потерь...")
        df['total_financial_loss'] = df.apply(
            lambda row: self.calculate_total_loss(
                row['base_financial_loss'],
                row['detection_time'],
                row['company_size'],
                row['intensity_multiplier']
            ),
            axis=1
        )

        logger.info(f"Средние потери: ${df['total_financial_loss'].mean():,.0f}")
        logger.info(f"Медианные потери: ${df['total_financial_loss'].median():,.0f}")
        logger.info(f"Макс потери: ${df['total_financial_loss'].max():,.0f}")

        return df

    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Статистика по финансовым потерям"""

        stats = {
            'total': df['total_financial_loss'].sum(),
            'mean': df['total_financial_loss'].mean(),
            'median': df['total_financial_loss'].median(),
            'std': df['total_financial_loss'].std(),
            'min': df['total_financial_loss'].min(),
            'max': df['total_financial_loss'].max()
        }

        # По типам атак
        label_col = self.config.data['target_column']
        stats['by_attack'] = df.groupby(label_col)['total_financial_loss'].agg([
            'count', 'mean', 'sum'
        ]).to_dict()

        return stats


if __name__ == "__main__":

    config = Config("configs/full.yaml")

    loader = DataLoader(config)
    data = loader.load(use_sample=True, sample_size=5000)

    engineer = FeatureEngineer(config)
    data = engineer.engineer(data)

    labeler = FinancialLabeler(config)
    data = labeler.add_financial_labels(data)

    stats = labeler.get_statistics(data)

    print(f"Общая сумма: ${stats['total']:,.0f}")
    print(f"Средние потери: ${stats['mean']:,.0f}")