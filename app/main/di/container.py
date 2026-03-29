from app.application.interactors.assess_financial_risk import (
    AssessFinancialRiskInteractor,
)
from app.application.interactors.load_dataset import LoadDatasetInteractor
from app.application.interactors.train_classifier import TrainClassifierInteractor
from app.application.interactors.train_regressor import TrainRegressorInteractor
from app.infrastructure.config import Config
from app.infrastructure.data.csv_loader import CSVDataLoader
from app.infrastructure.data.preprocessor import RobustPreprocessor
from app.infrastructure.features.engineer import FeatureEngineer
from app.infrastructure.financial.calculator import RiskAssessor, RiskCalculator
from app.infrastructure.models.lightgbm_model import LightGBMModel
from app.infrastructure.models.random_forest import RandomForestModel
from app.infrastructure.models.xgboost_model import XGBoostModel
from app.infrastructure.repositories.model_repository import FileModelRepository
from app.presentation.visualizer import Visualizer


class Container:
    """IoC-контейнер - собирает все зависимости в одном месте (DIP)."""

    def __init__(self, config: Config) -> None:
        self._config = config

        self._assessor = RiskAssessor(config.financial)
        self._calculator = RiskCalculator(self._assessor)
        self._loader = CSVDataLoader(config.data.path)
        self._engineer = FeatureEngineer()
        self._preprocessor = RobustPreprocessor()
        self._repository = FileModelRepository(config.models.save_path)
        self._visualizer = Visualizer(config.results.path)

    def load_dataset(self) -> LoadDatasetInteractor:
        """Интерактор загрузки и обогащения датасета."""
        return LoadDatasetInteractor(
            data_loader=self._loader,
            feature_engineer=self._engineer,
            financial_calculator=self._calculator,
        )

    def train_classifier(self, model_name: str) -> TrainClassifierInteractor:
        """Интерактор обучения классификатора."""
        models = {
            "random_forest": RandomForestModel("classification"),
            "xgboost": XGBoostModel("classification", {"device": "cuda"}),
            "lightgbm": LightGBMModel("classification", {"device": "gpu"}),
        }
        model = models.get(model_name)
        if model is None:
            raise ValueError(
                f"Неизвестный классификатор: '{model_name}'. "
                f"Доступные: {list(models.keys())}"
            )
        return TrainClassifierInteractor(model=model, repository=self._repository)

    def train_regressor(self) -> TrainRegressorInteractor:
        """Интерактор обучения регрессора финансовых потерь (XGBoost)."""
        return TrainRegressorInteractor(
            model=XGBoostModel("regression", {"device": "cuda"}),
            repository=self._repository,
        )

    def assess_risk(self) -> AssessFinancialRiskInteractor:
        """Интерактор оценки риска одной атаки."""
        return AssessFinancialRiskInteractor(
            calculator=self._calculator,
            assessor=self._assessor,
        )

    def preprocessor(self) -> RobustPreprocessor:
        return self._preprocessor

    def visualizer(self) -> Visualizer:
        return self._visualizer
