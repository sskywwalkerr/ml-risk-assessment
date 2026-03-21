import numpy as np
import pandas as pd

from app.application.interfaces.financial import (
    IFinancialCalculator,
    IRiskAssessor,
    SingleRiskRequest,
)
from app.domain import FinancialRisk, RiskLevel
from app.infrastructure.financial.cost_model import (
    BASE_COSTS,
    DETECTION_TIME_MULTIPLIER,
    LOSS_WEIGHTS,
)


class RiskAssessor(IRiskAssessor):
    """Определяет уровень риска по итоговым потерям."""

    _THRESHOLDS: list[tuple[float, RiskLevel]] = [
        (1_000_000.0, RiskLevel.CRITICAL),
        (100_000.0, RiskLevel.HIGH),
        (10_000.0, RiskLevel.MEDIUM),
    ]

    def assess(self, total_loss: float) -> RiskLevel:
        """Возвращает уровень риска."""
        for threshold, level in self._THRESHOLDS:
            if total_loss >= threshold:
                return level
        return RiskLevel.LOW


class RiskCalculator(IFinancialCalculator):
    """
    Реализация формулы ожидаемого годового убытка (ALE).
    ALE = Вероятность * Базовая_Стоимость * Интенсивность * Время_Обнаружения
    """

    def __init__(self, assessor: IRiskAssessor) -> None:
        self._assessor = assessor
        # Средняя стоимость атаки для случаев, когда тип атаки неизвестен
        self._default_cost = float(np.mean([v for v in BASE_COSTS.values() if v > 0]))

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет финансовые колонки для всего DataFrame."""
        df = df.copy()
        # базовую стоимость и множитель задержки к типу атаки (label)
        base = df["label"].map(BASE_COSTS).fillna(self._default_cost)
        det_mult = df["label"].map(DETECTION_TIME_MULTIPLIER).fillna(1.0)

        # динамическая интенсивность на основе сетевых признаков
        intensity = self._compute_intensity(df)

        # Разбитие ущерба на категории (прямой, косвенный, репутация, штрафы)
        df["base_financial_loss"] = base
        df["intensity_multiplier"] = intensity
        df["direct_loss"] = base * LOSS_WEIGHTS["direct"]
        df["indirect_loss"] = base * LOSS_WEIGHTS["indirect"]
        df["reputation_loss"] = base * LOSS_WEIGHTS["reputation"]
        df["regulatory_fine"] = base * LOSS_WEIGHTS["regulatory"]

        # ALE без вероятности (P=1.0 в batch-режиме - предполагаем что атака произошла)
        df["total_financial_loss"] = (base * intensity * det_mult).clip(
            lower=0.0, upper=10_000_000.0
        )

        return df

    def calculate_single(self, request: SingleRiskRequest) -> FinancialRisk:
        """Рассчитывает ALE для одной атаки с вероятностью из классификатора."""
        base = BASE_COSTS.get(request.attack_type, self._default_cost)
        det_mult = DETECTION_TIME_MULTIPLIER.get(request.attack_type, 1.0)

        # Доли ущерба
        direct = base * LOSS_WEIGHTS["direct"]
        indirect = base * LOSS_WEIGHTS["indirect"]
        reputation = base * LOSS_WEIGHTS["reputation"]
        regulatory = base * LOSS_WEIGHTS["regulatory"]

        total = base * request.intensity * det_mult * request.probability

        return FinancialRisk(
            attack_type=request.attack_type,
            direct_loss=direct,
            indirect_loss=indirect,
            reputation_loss=reputation,
            regulatory_fine=regulatory,
            intensity_multiplier=request.intensity * det_mult,
            probability=request.probability,
            risk_level=self._assessor.assess(total),  # Авто-оценка уровня риска
        )

    @staticmethod
    def _compute_intensity(df: pd.DataFrame) -> pd.Series:
        """
        Превращает сетевые метрики (rate, header_length) в коэффициент [0.5, 2.0].
        Помогает отличить 'ленивую' атаку от агрессивной.
        """
        # Преобразование в числа и очистка от мусора/отрицательных значений
        rate = (
            pd.to_numeric(
                df.get("rate", pd.Series(0.0, index=df.index)), errors="coerce"
            )
            .fillna(0)
            .clip(0)
        )
        header = (
            pd.to_numeric(
                df.get("header_length", pd.Series(0.0, index=df.index)), errors="coerce"
            )
            .fillna(0)
            .clip(0)
        )

        # вычисление 99й перцентиль, отсечь редкие аномальные выбросы и определить «нормальный максимум»
        p99_rate = (
            rate.quantile(0.99) * 1e-8
        )  # масштабирование, чтобы привести огромные значения к рабочему диапазону.
        p99_header = header.quantile(0.99) * 1e-8

        # Весовая формула: скорость важнее (60%), длина заголовков дополняет (40%)
        intensity = (rate / p99_rate) * 0.6 + (header / p99_header) * 0.4

        # Ограничение множителя, чтобы он не обнулял риск и не раздувал его до бесконечности
        return intensity.clip(lower=0.5, upper=2.0)
