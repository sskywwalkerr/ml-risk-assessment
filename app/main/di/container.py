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
    """IoC-контейнер — собирает все зависимости в одном месте (DIP).

    Единственное место, где знает о конкретных реализациях.
    Все модули выше работают только с абстракциями.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

        # Финансовый модуль — параметры из config.yaml, не из хардкода
        self._assessor = RiskAssessor(config.financial)
        self._calculator = RiskCalculator(config.financial, self._assessor)

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
        """Интерактор обучения классификатора.

        Параметры моделей загружаются из config.yaml.
        """
        cfg = self._config.models
        models = {
            "random_forest": RandomForestModel(
                "classification",
                cfg.get("random_forest", {}).get("classification", {}),
            ),
            "xgboost": XGBoostModel(
                "classification",
                cfg.get("xgboost", {}).get("classification", {}),
            ),
            "lightgbm": LightGBMModel(
                "classification",
                cfg.get("lightgbm", {}).get("classification", {}),
            ),
        }
        model = models.get(model_name)
        if model is None:
            raise ValueError(
                f"Неизвестный классификатор: '{model_name}'. "
                f"Доступные: {list(models.keys())}"
            )
        return TrainClassifierInteractor(model=model, repository=self._repository)

    def train_regressor(self, model_name: str) -> TrainRegressorInteractor:
        """Интерактор обучения регрессора финансовых потерь."""
        cfg = self._config.models
        models = {
            "xgboost_regressor": XGBoostModel(
                "regression",
                cfg.get("xgboost", {}).get("regression", {}),
            ),
            "lightgbm_regressor": LightGBMModel(
                "regression",
                cfg.get("lightgbm", {}).get("regression", {}),
            ),
            "random_forest_regressor": RandomForestModel(
                "regression",
                cfg.get("random_forest", {}).get("regression", {}),
            ),
        }
        model = models.get(model_name)
        if model is None:
            raise ValueError(f"Неизвестный регрессор: '{model_name}'")
        return TrainRegressorInteractor(model=model, repository=self._repository)

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
