import pandas as pd
from pathlib import Path
from typing import Optional

from src.interfaces.base import IDataLoader
from src.utils.config import Config


def fix_string_encoding(text):
    """Исправление некорректных символов в строках"""
    if isinstance(text, str):
        replacements = {
            '�': '-',
            '–': '-',
            '—': '-',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
            '…': '...',
            '\x96': '-',
            '\x97': '-',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()
    return text


def normalize_labels(df, label_col='Label'):
    label_mapping = {
        'BENIGN': 'BENIGN',
        'benign': 'BENIGN',
        '-B-E-N-I-G-N-': 'BENIGN',
        'DoS Hulk': 'DoS Hulk',
        'DoS-Hulk': 'DoS Hulk',
        'DDoS': 'DDoS',
        'PortScan': 'PortScan',
        'FTP-Patator': 'FTP-Patator',
        'SSH-Patator': 'SSH-Patator',
        'Web Attack - Brute Force': 'Web Attack Brute Force',
        'Web Attack - XSS': 'Web Attack XSS',
        'Web Attack - Sql Injection': 'Web Attack Sql Injection',
        '-W-e-b- -A-t-t-a-c-k- --- -B-r-u-t-e- -F-o-r-c-e-': 'Web Attack Brute Force',
        '-W-e-b- -A-t-t-a-c-k- --- -X-S-S-': 'Web Attack XSS',
        '-D-o-S- -H-u-l-k-': 'DoS Hulk',
        '-D-D-o-S-': 'DDoS',
        '-P-o-r-t-S-c-a-n-': 'PortScan',
        '-B-o-t-': 'Bot',
        'Bot': 'Bot',
        'Heartbleed': 'Heartbleed',
        '-H-e-a-r-t-b-l-e-e-d-': 'Heartbleed',
        'DoS GoldenEye': 'DoS GoldenEye',
        '-D-o-S- -G-o-l-d-e-n-E-y-e-': 'DoS GoldenEye',
        'DoS Slowhttptest': 'DoS Slowhttptest',
        '-D-o-S- -S-l-o-w-h-t-t-p-t-e-s-t-': 'DoS Slowhttptest',
        'DoS slowloris': 'DoS Slowloris',
        'DoS Slowloris': 'DoS Slowloris',
        '-D-o-S- -s-l-o-w-l-o-r-i-s-': 'DoS Slowloris',
        'Infiltration': 'Infiltration'
    }
    df[label_col] = df[label_col].astype(str).map(label_mapping).fillna('BENIGN')
    return df

class DataLoader(IDataLoader):
    def __init__(self, config: Config):
        self.config = config
        self.data_path = Path(config.paths['data_raw'])
        self.target_column = config.data['target_column']

    def load_single_file(self, filepath: Path) -> pd.DataFrame:
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(
                        filepath,
                        encoding=encoding,
                        skipinitialspace=True,
                        low_memory=False
                    )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None
            df.columns = [fix_string_encoding(col).strip() for col in df.columns]
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].apply(fix_string_encoding)
            return df
        except Exception as e:
            print(f"Ошибка при загрузке {filepath}: {e}")
            return None

    def load_all_files(self) -> pd.DataFrame:
        csv_files = sorted(self.data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"CSV файлы не найдены в {self.data_path}")
        dataframes = []
        for filepath in csv_files:
            df = self.load_single_file(filepath)
            if df is not None:
                dataframes.append(df)
        if not dataframes:
            raise ValueError("Не удалось загрузить ни одного файла")
        return pd.concat(dataframes, ignore_index=True)

    def load(self, use_sample: bool = False, sample_size: Optional[int] = None) -> pd.DataFrame:
        df = self.load_all_files()
        initial_size = len(df)
        df = df.dropna(how='all')
        if self.target_column not in df.columns:
            raise ValueError(f"Колонка '{self.target_column}' не найдена")
        df = normalize_labels(df, self.target_column)
        if use_sample:
            size = sample_size or self.config.data.get('sample_size')
            if size and size < len(df):
                df = df.sample(n=size, random_state=42)
        print(f"Загружено: {len(df):,} строк, {len(df.columns)} колонок")
        return df
