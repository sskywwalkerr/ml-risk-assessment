from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class AppConfig:
    debug: bool = False


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Настройки предобработки данных."""

    test_size: float = 0.2
    val_size: float = 0.125  # 12.5% от оставшихся -> итого 70/10/20
    random_state: int = 42
    nan_threshold: float = 0.5  # Удалять строки где >50% числовых колонок - NaN


@dataclass(frozen=True, slots=True)
class FinancialConfig:
    """Финансовые параметры модели оценки риска."""

    currency: str = "RUB"
    usd_to_rub: float = 92.66
    base_costs_rub: dict[str, float] = field(default_factory=dict)
    loss_weights: dict[str, float] = field(default_factory=dict)
    detection_time_multiplier: dict[str, float] = field(default_factory=dict)
    max_loss_rub: float = 10_000_000.0
    risk_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    """Настройки моделей машинного обучения."""

    save_path: str = "models"
    _params: dict[str, Any] = field(default_factory=dict)

    def get(self, model_name: str, default: Any = None) -> Any:
        """Возвращает параметры модели по имени."""
        return self._params.get(model_name, default or {})


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Настройки датасета CICIoT2023."""

    path: str = "data/raw"
    sample_size: int | None = None
    target_column: str = "label"


@dataclass(frozen=True, slots=True)
class ResultsConfig:
    """Настройки сохранения результатов и графиков."""

    path: str = "results"
    dpi: int = 150
    save_feature_importance: bool = True
    save_confusion_matrix: bool = True
    save_regression_analysis: bool = True


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
    def from_yaml(path: str = "app/main/config.yaml") -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Конфиг не найден: {path}")

        with open(config_path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        models_raw = raw.get("models", {})
        models_params = {k: v for k, v in models_raw.items() if k != "save_path"}

        results_raw = raw.get("results", {})

        return Config(
            app=AppConfig(**raw.get("app", {})),
            data=DataConfig(**raw.get("data", {})),
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            models=ModelsConfig(
                save_path=models_raw.get("save_path", "models"),
                _params=models_params,
            ),
            financial=FinancialConfig(**raw.get("financial", {})),
            results=ResultsConfig(
                path=results_raw.get("path", "results"),
                dpi=results_raw.get("dpi", 150),
                save_feature_importance=results_raw.get(
                    "save_feature_importance", True
                ),
                save_confusion_matrix=results_raw.get("save_confusion_matrix", True),
                save_regression_analysis=results_raw.get(
                    "save_regression_analysis", True
                ),
            ),
        )
