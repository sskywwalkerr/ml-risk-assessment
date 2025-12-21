import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = "configs/full.yaml"):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Конфиг не найден: {config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def to_dict(self) -> Dict:
        return self._config.copy()

    @property
    def paths(self):
        return self._config.get('paths', {})

    @property
    def data(self):
        return self._config.get('data', {})

    @property
    def models(self):
        return self._config.get('models', {})

    @property
    def gpu(self):
        return self._config.get('gpu', {})
