from dataclasses import dataclass

from app.domain.enums import AttackCategory


@dataclass(frozen=True, slots=True)
class Attack:
    """Результат классификации атаки."""

    attack_type: str
    category: AttackCategory
    probability: float
    is_benign: bool = False

    @property
    def severity_weight(self) -> float:
        """Числовой вес категории для финансовой модели."""
        weights = {
            AttackCategory.BENIGN: 0.0,
            AttackCategory.RECON: 0.2,
            AttackCategory.SPOOFING: 0.4,
            AttackCategory.WEB_BASED: 0.6,
            AttackCategory.BRUTE_FORCE: 0.7,
            AttackCategory.DOS: 0.7,
            AttackCategory.DDOS: 0.8,
            AttackCategory.MIRAI: 1.0,
        }
        return weights.get(self.category, 0.5)
