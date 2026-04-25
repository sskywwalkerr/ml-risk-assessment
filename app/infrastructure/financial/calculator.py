import numpy as np
import pandas as pd

from app.application.interfaces.financial import (
    IFinancialCalculator,
    IRiskAssessor,
    SingleRiskRequest,
)
from app.domain import FinancialRisk, RiskLevel
from app.infrastructure.config import FinancialConfig


class RiskAssessor(IRiskAssessor):
    """Определяет уровень риска по итоговым потерям."""

    def __init__(self, config: FinancialConfig) -> None:
        self._thresholds: list[tuple[float, RiskLevel]] = [
            (config.risk_thresholds["critical"], RiskLevel.CRITICAL),
            (config.risk_thresholds["high"], RiskLevel.HIGH),
            (config.risk_thresholds["medium"], RiskLevel.MEDIUM),
        ]

    def assess(self, total_loss: float) -> RiskLevel:
        """Возвращает уровень риска: LOW / MEDIUM / HIGH / CRITICAL."""
        for threshold, level in self._thresholds:
            if total_loss >= threshold:
                return level
        return RiskLevel.LOW


class RiskCalculator(IFinancialCalculator):
    """
    Реализация формулы ожидаемого годового убытка (ALE).
    ALE = Вероятность * Базовая_Стоимость * Интенсивность * Время_Обнаружения
    """

    def __init__(self, config: FinancialConfig, assessor: IRiskAssessor) -> None:
        self._config = config
        self._assessor = assessor
        self._base_costs = config.base_costs_rub
        self._loss_weights = config.loss_weights
        self._detection_multipliers = config.detection_time_multiplier
        self._max_loss = config.max_loss_rub

        # Средняя стоимость атаки для неизвестных типов
        non_zero = [v for v in self._base_costs.values() if v > 0]
        self._default_cost = float(np.mean(non_zero)) if non_zero else 0.0

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет финансовые колонки для всего DataFrame."""
        df = df.copy()
        # C_base - базовая стоимость по типу атаки из конфига
        base = df["label"].map(self._base_costs).fillna(self._default_cost)

        # K_detection - коэффициент сложности обнаружения из конфига
        det_mult = df["label"].map(self._detection_multipliers).fillna(1.0)

        # M_intensity - динамический множитель по характеристикам трафика
        intensity = self._compute_intensity(df)

        # Разбитие ущерба на категории (прямой, косвенный, репутация, штрафы)
        df["base_financial_loss"] = base
        df["intensity_multiplier"] = intensity
        df["direct_loss"] = base * self._loss_weights["direct"]
        df["indirect_loss"] = base * self._loss_weights["indirect"]
        df["reputation_loss"] = base * self._loss_weights["reputation"]
        df["regulatory_fine"] = base * self._loss_weights["regulatory"]

        # Итоговый ущерб L = C_base * M_intensity * K_detection
        # P = 1.0 в batch-режиме (предполагаем факт атаки)
        df["total_financial_loss"] = (base * intensity * det_mult).clip(
            lower=0.0, upper=self._max_loss
        )

        return df

    def calculate_single(self, request: SingleRiskRequest) -> FinancialRisk:
        """Рассчитывает ALE для одной атаки с вероятностью из классификатора."""
        base = self._base_costs.get(request.attack_type, self._default_cost)
        det_mult = self._detection_multipliers.get(request.attack_type, 1.0)

        # Доли ущерба
        direct = base * self._loss_weights["direct"]
        indirect = base * self._loss_weights["indirect"]
        reputation = base * self._loss_weights["reputation"]
        regulatory = base * self._loss_weights["regulatory"]

        total = base * request.intensity * det_mult * request.probability

        return FinancialRisk(
            attack_type=request.attack_type,
            direct_loss=direct,
            indirect_loss=indirect,
            reputation_loss=reputation,
            regulatory_fine=regulatory,
            intensity_multiplier=request.intensity * det_mult,
            probability=request.probability,
            risk_level=self._assessor.assess(total),
        )

    @staticmethod
    def _compute_intensity(df: pd.DataFrame) -> pd.Series:
        """M_intensity из характеристик трафика.
        Формула: 0.6 * norm(rate) + 0.4 * norm(header_length)
        Диапазон: [0.5, 2.0] - от фонового трафика до пиковой нагрузки.
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
        p99_rate = max(
            float(rate.quantile(0.99)), 1e-8
        )  # масштабирование, чтобы привести огромные значения к рабочему диапазону.
        p99_header = max(float(header.quantile(0.99)), 1e-8)

        # Весовая формула: скорость важнее (60%), длина заголовков дополняет (40%)
        intensity = (rate / p99_rate) * 0.6 + (header / p99_header) * 0.4

        # Ограничение множителя, чтобы он не обнулял риск и не раздувал его до бесконечности
        return intensity.clip(lower=0.5, upper=2.0)
