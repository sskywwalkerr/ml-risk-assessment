from dataclasses import dataclass
from app.domain.enums import RiskLevel


@dataclass(frozen=True)
class FinancialRisk:
    """Финансовая оценка риска одной атаки."""
    attack_type: str

    # Компоненты потерь (USD)
    direct_loss: float  # Восстановление систем
    indirect_loss: float  # Простой
    reputation_loss: float  # Долгосрочный ущерб репутации
    regulatory_fine: float  # Штрафы

    intensity_multiplier: float  # Нормализованный множитель интенсивности
    probability: float  # Вероятность

    risk_level: RiskLevel

    @property
    def total_loss(self) -> float:
        """Ожидание общей потери."""
        base = self.direct_loss + self.indirect_loss + self.reputation_loss + self.regulatory_fine
        return round(base * self.intensity_multiplier * self.probability, 2)

    @property
    def impact(self) -> float:
        """Максимально возможный ущерб"""
        return self.direct_loss + self.indirect_loss + self.reputation_loss + self.regulatory_fine
