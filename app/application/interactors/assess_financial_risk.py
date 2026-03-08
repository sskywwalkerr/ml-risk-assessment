from dataclasses import dataclass

from app.application.interfaces.financial import (
    IFinancialCalculator,
    IRiskAssessor,
    SingleRiskRequest,
)
from app.domain.entities.financial_risk import FinancialRisk


@dataclass(frozen=True, slots=True)
class AssessFinancialRiskRequest:
    """Параметры для оценки финансового риска одной атаки."""

    attack_type: str
    probability: float
    intensity: float


@dataclass(frozen=True, slots=True)
class AssessFinancialRiskResponse:
    """Результат оценки финансового риска."""

    risk: FinancialRisk
    summary: dict


class AssessFinancialRiskInteractor:
    """Оценивает финансовый риск для конкретного типа атаки."""

    def __init__(
        self,
        calculator: IFinancialCalculator,
        assessor: IRiskAssessor,
    ) -> None:
        self._calculator = calculator
        self._assessor = assessor

    def __call__(
        self, request: AssessFinancialRiskRequest
    ) -> AssessFinancialRiskResponse:
        risk = self._calculator.calculate_single(
            SingleRiskRequest(
                attack_type=request.attack_type,
                probability=request.probability,
                intensity=request.intensity,
            )
        )
        return AssessFinancialRiskResponse(
            risk=risk,
            summary={
                "attack_type": risk.attack_type,
                "total_loss": risk.total_loss,
                "risk_level": risk.risk_level,
                "impact": risk.impact,
                "probability": risk.probability,
            },
        )
