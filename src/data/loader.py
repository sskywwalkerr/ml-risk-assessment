import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


class DataLoader:
    def __init__(self, config: Config):
        """
        Инициализация

        Args:
            config: объект конфигурации
        """
        self.config = config
        self.data_path = Path(config.paths['data_raw'])
        self.target_column = config.data['target_column']

    def load_single_file(self, filepath: Path) -> pd.DataFrame:
        """
        Загрузка одного CSV файла

        Args:
            filepath: путь к файлу

        Returns:
            DataFrame
        """
        try:
            df = pd.read_csv(
                filepath,
                encoding='utf-8',
                skipinitialspace=True,
                low_memory=False
            )

            # Очистка названий колонок
            df.columns = df.columns.str.strip()

            logger.info(f"{filepath.name}: {len(df):,} строк")
            return df

        except Exception as e:
            logger.error(f"Ошибка загрузки {filepath.name}: {e}")
            return None

    def load_all_files(self) -> pd.DataFrame:
        """
        Загрузка и объединение всех CSV файлов

        Returns:
            Объединенный DataFrame
        """
        csv_files = sorted(self.data_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"CSV файлы не найдены в {self.data_path}")

        logger.info(f"Найдено файлов: {len(csv_files)}\n")

        # Загружаем файлы с прогресс-баром
        dataframes = []
        for filepath in tqdm(csv_files, desc="Загрузка файлов"):
            df = self.load_single_file(filepath)
            if df is not None:
                dataframes.append(df)

        if not dataframes:
            raise ValueError("Не удалось загрузить ни одного файла")

        combined_df = pd.concat(dataframes, ignore_index=True)

        logger.info(f"Размер: {len(combined_df):,} строк  {len(combined_df.columns)} колонок")
        logger.info(f"Память: {combined_df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

        return combined_df

    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Получить информацию о датасете

        Args:
            df: DataFrame

        Returns:
            Словарь с информацией
        """
        info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 ** 2,
            'dtypes': df.dtypes.value_counts().to_dict()
        }

        # Распределение классов
        if self.target_column in df.columns:
            info['class_distribution'] = df[self.target_column].value_counts().to_dict()

        return info

    def print_info(self, df: pd.DataFrame):
        """
        Вывод информации о датасете

        Args:
            df: DataFrame
        """
        info = self.get_dataset_info(df)

        logger.info(f"ИНФОРМАЦИЯ О ДАТАСЕТЕ")

        logger.info(f"Размер:")
        logger.info(f"Строк: {info['total_rows']:,}")
        logger.info(f"Колонок: {info['total_columns']}")
        logger.info(f"Пропусков: {info['missing_values']:,}")
        logger.info(f"Память: {info['memory_mb']:.1f} MB")

        if 'class_distribution' in info:
            logger.info(f"\nРаспределение классов:")
            for class_name, count in sorted(
                info['class_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / info['total_rows']) * 100
                bar = int(percentage / 2) * "█"
                logger.info(f"   {class_name:30s} {count:8,} ({percentage:5.2f}%) {bar}")

    def load(
        self,
        use_sample: bool = False,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Главный метод загрузки

        Args:
            use_sample: использовать ли выборку
            sample_size: размер выборки (если use_sample=True)

        Returns:
            DataFrame с данными
        """
        # Загрузка всех файлов
        df = self.load_all_files()

        # Базовая очистка
        initial_size = len(df)
        df = df.dropna(how='all')  # Удаляем полностью пустые строки

        if len(df) < initial_size:
            logger.info(f"Удалено пустых строк: {initial_size - len(df):,}")

        if self.target_column not in df.columns:
            raise ValueError(f"Колонка '{self.target_column}' не найдена")

        if use_sample:
            size = sample_size or self.config.data.get('sample_size')
            if size and size < len(df):
                logger.info(f"\nСоздание sample: {size:,} строк")
                df = df.sample(n=size, random_state=42)

        self.print_info(df)

        # Сохранение (опционально)
        save_format = self.config.data.get('format')
        if save_format == 'parquet':
            self._save_parquet(df, 'combined_dataset.parquet')
        elif save_format == 'csv':
            self._save_csv(df, 'combined_dataset.csv')

        return df

    def _save_parquet(self, df: pd.DataFrame, filename: str):
        """Сохранение в Parquet формате"""
        output_path = Path(self.config.paths['data_processed'])
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / filename
        df.to_parquet(filepath, compression='snappy', index=False)

        file_size = filepath.stat().st_size / 1024 ** 2
        logger.info(f"\nСохранено в Parquet: {filepath}")
        logger.info(f"Размер: {file_size:.1f} MB")

    def _save_csv(self, df: pd.DataFrame, filename: str):
        """Сохранение в CSV формате"""
        output_path = Path(self.config.paths['data_processed'])
        output_path.mkdir(parents=True, exist_ok=True)

        if not filename.endswith('.csv'):
            filename += '.csv'

        filepath = output_path / filename
        df.to_csv(filepath, index=False, encoding='utf-8')

        file_size = filepath.stat().st_size / 1024 ** 2
        logger.info(f"\nСохранено в CSV: {filepath}")
        logger.info(f"Размер: {file_size:.1f} MB")


if __name__ == "__main__":
    config = Config("configs/full.yaml")

    loader = DataLoader(config)
    data = loader.load(use_sample=True, sample_size=10000)

    print(f"\nЗагрузка завершена!")
    print(f"Размер: {data.shape}")