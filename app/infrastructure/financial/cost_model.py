"""Базовые стоимости потерь по категориям атак CICIoT2023.

Источники:
  [1] IBM Cost of Data Breach Report 2024
      https://ibm.com/reports/data-breach
  [2] Verizon Data Breach Investigations Report 2024
      https://verizon.com/business/resources/reports/dbir
  [3] Zayo DDoS Insights Report 2023
      https://zayo.com/resources/ddos-insights-report
  [4] Sophos State of Ransomware 2024
      https://sophos.com/en-us/whitepaper/state-of-ransomware

Структура потерь (% от total, методология IBM [1]):
  direct_loss      40% — восстановление систем, IR-команда
  indirect_loss    30% — простой сервисов, потеря клиентов
  reputation_loss  15% — долгосрочный репутационный ущерб
  regulatory_fine  15% — штрафы GDPR (до 4% оборота) / HIPAA
"""

# Базовые стоимости по категориям атак (USD)
BASE_COSTS: dict[str, float] = {
    # Benign - нет потерь
    "BenignTraffic": 0.0,
    # DDoS: $6,000/мин × 68 мин средней атаки [3]
    "DDoS": 408_000.0,
    # DoS: [3] H1 2024 average
    "DoS": 270_000.0,
    # Mirai/Botnet: Sophos 2024 ransomware/botnet average [4]
    "Mirai": 5_130_000.0,
    # BruteForce: IBM 2024 — stolen credentials vector [1]
    "BruteForce": 4_810_000.0,
    # Web-based: IBM 2024 — web attack vector average [1]
    # $173/запись × ~3,750 средних записей
    "Web-based": 650_000.0,
    # Recon: прямой ущерб минимален — только разведка
    "Recon": 50_000.0,
    # Spoofing: Verizon DBIR 2024 — MITM/spoofing incidents [2]
    "Spoofing": 1_200_000.0,
}

# Доли компонентов потерь (IBM methodology [1])
LOSS_WEIGHTS: dict[str, float] = {
    "direct": 0.40,
    "indirect": 0.30,
    "reputation": 0.15,
    "regulatory": 0.15,
}

# Множители времени обнаружения (IBM [1]: среднее 292 дня для BruteForce)
DETECTION_TIME_MULTIPLIER: dict[str, float] = {
    "BenignTraffic": 1.0,
    "Recon": 1.1,  # Быстро обнаруживается
    "DDoS": 1.2,  # Очевидно, но требует реакции
    "DoS": 1.2,
    "Spoofing": 1.4,  # Сложнее обнаружить
    "Web-based": 1.5,
    "BruteForce": 1.8,  # IBM: 292 дня среднее
    "Mirai": 2.0,  # Долго остаётся незамеченным
}
