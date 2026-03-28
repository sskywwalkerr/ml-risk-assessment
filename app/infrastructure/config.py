from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class AppConfig:
    debug: bool = False


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Настройки предобработки."""

    test_size: float = 0.2
    val_size: float = 0.125
    random_state: int = 42
    nan_threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class FinancialConfig:
    currency: str = "RUB"
    usd_to_rub: float = 92.66
    base_costs_rub: dict[str, float] = field(default_factory=dict)
    loss_weights: dict[str, float] = field(default_factory=dict)
    detection_time_multiplier: dict[str, float] = field(default_factory=dict)
    max_loss_rub: float = 926_600_000.0
    risk_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Настройки датасета CICIoT2023."""

    path: str = "data/raw"
    sample_size: int | None = None
    target_column: str = "label"


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    """Настройки сохранения моделей."""

    save_path: str = "models"


@dataclass(frozen=True, slots=True)
class ResultsConfig:
    """Настройки сохранения результатов."""

    path: str = "results"


@dataclass(frozen=True, slots=True)
class Config:
    """Корневой конфиг проекта."""

    data: DataConfig
    app: AppConfig = field(default_factory=AppConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    financial: FinancialConfig = field(default_factory=FinancialConfig)
    results: ResultsConfig = field(default_factory=ResultsConfig)

    @staticmethod
    def from_yaml(path: str = "configs/config.yaml") -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Конфиг не найден: {path}")

        with open(config_path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        return Config(
            app=AppConfig(**raw.get("app", {})),
            data=DataConfig(**raw.get("data", {})),
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            models=ModelsConfig(
                save_path=raw.get("models", {}).get("save_path", "models")
            ),
            financial=FinancialConfig(**raw.get("financial", {})),
            results=ResultsConfig(path=raw.get("results", {}).get("path", "results")),
        )
