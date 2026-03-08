from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from app.domain.entities.financial_risk import FinancialRisk
from app.domain.enums import RiskLevel


@dataclass(frozen=True, slots=True)
class SingleRiskRequest:
    """Параметры для расчета риска одной атаки."""

    attack_type: str
    probability: float
    intensity: float


class IFinancialCalculator(ABC):
    """Интерфейс для расчета финансового ущерба от инцидента ИБ."""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет финансовые колонки в DataFrame."""
        ...

    @abstractmethod
    def calculate_single(self, request: SingleRiskRequest) -> FinancialRisk:
        """Рассчитывает финансовый риск для одной записи."""
        ...


class IRiskAssessor(ABC):
    """Интерфейс для определения уровня риска по сумме потерь."""

    @abstractmethod
    def assess(self, total_loss: float) -> RiskLevel:
        """Возвращает уровень риска: LOW / MEDIUM / HIGH / CRITICAL."""
        ...
