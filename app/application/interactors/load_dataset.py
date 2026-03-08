from dataclasses import dataclass

import pandas as pd

from app.application.interfaces.data_loader import IDataLoader, LoadDataRequest
from app.application.interfaces.feature_engineer import IFeatureEngineer
from app.application.interfaces.financial import IFinancialCalculator


@dataclass(frozen=True, slots=True)
class LoadDatasetRequest:
    """Параметры запуска интерактора загрузки датасета."""

    sample_size: int | None = None


@dataclass(frozen=True, slots=True)
class LoadDatasetResponse:
    """Результат загрузки - обогащённый DataFrame и статистика."""

    df: pd.DataFrame
    n_rows: int
    n_cols: int
    label_distribution: dict


class LoadDatasetInteractor:
    """Загружает CICIoT2023, создаёт признаки и добавляет финансовые метки."""

    def __init__(
        self,
        data_loader: IDataLoader,
        feature_engineer: IFeatureEngineer,
        financial_calculator: IFinancialCalculator,
    ) -> None:
        self._loader = data_loader
        self._engineer = feature_engineer
        self._calculator = financial_calculator

    def __call__(self, request: LoadDatasetRequest) -> LoadDatasetResponse:
        df = self._loader.load(LoadDataRequest(sample_size=request.sample_size))
        df = self._engineer.engineer(df)
        df = self._calculator.calculate(df)

        return LoadDatasetResponse(
            df=df,
            n_rows=len(df),
            n_cols=len(df.columns),
            label_distribution=df["label"].value_counts().to_dict(),
        )
