BASE_COSTS: dict[str, float] = {
    # BenignTraffic: потерь нет
    "BenignTraffic": 0.0,
    # DDoS: $234,000 * 92.66 = 21,682,440 ₽
    # Zayo DDoS Insights Report 2024:
    # $6000/мин, среднее за 39 минут = $234000
    # URL: https://www.zayo.com/newsroom/ai-adoption-and-iot-proliferation-fuel-82-spike-in-ddos-attacks-in-2024-according-to-zayo/
    "DDoS": 21_682_440.0,
    # DoS: $4,410,000 * 92.66 = 408,630,600 ₽
    # IBM Cost of a Data Breach Report 2025
    # URL: https://www.ibm.com/downloads/documents/us-en/131cf87b20b31c91
    "DoS": 25_018_200.0,
    # Mirai/Botnet: $4,910,000 * 92.66 = 454,958,600 Р
    # IBM 2024 глобальный средний ущерб $4.88M
    # веб-атаки составляют ~13% инцидентов(Verizon DBIR 2024) доля от среднего: $4.88M * 0.13 ≈ $635,000 * 92.66 ≈ 58,800,000 ₽
    # URL: https://www.ibm.com/downloads/documents/us-en/131cf87b20b31c91
    "Mirai": 454_958_600.0,
    # BruteForce: $4,670,000 * 92.66 = 432,722,200 ₽
    # IBM Cost of a Data Breach Report 2025
    # В среднем 4.67M миллиона долларов, на выявление и устранение последствий уходит 246 дня
    # URL: https://wp.table.media/wp-content/uploads/2024/07/30132828/Cost-of-a-Data-Breach-Report-2024.pdf
    "BruteForce": 432_722_200.0,
    # Web-based: 60,000,000 Р
    # IBM Cost of a Data Breach 2024 - глобальный средний ущерб $4.88M
    # Веб-атаки составляют 13.3% от общего числа инцидентов (Verizon DBIR 2024)
    # $4.88M * 0.133 ≈ $650,000 - доля от среднего ущерба
    "Web-based": 60_000_000.0,
    # $50,000 * 92.66 = 4,633,000 ₽
    "Recon": 4_633_000.0,
    # Phishing/Spoofing : 373,200 Р за одну операцию данные ЦБ РФ
    # Отчет ФинЦЕРТ Банка России «Обзор операций, совершенных без согласия клиентов за 2025 год».
    # URL: https://www.cbr.ru/analytics/ib/operations_survey/2025/
    "Spoofing": 111_960_000.0,
}

# Доли компонентов потерь
# IBM Cost of a Data Breach Report 2025
# (Detection  Escalation, Notification, Lost Business, Post-breach Response)
LOSS_WEIGHTS: dict[str, float] = {
    "direct": 0.40,  # восстановление систем, IR-команда
    "indirect": 0.30,  # простой бизнеса, потеря дохода
    "reputation": 0.15,  # снижение доверия клиентов, отток
    "regulatory": 0.15,  # штрафы по ст. 13.11-13.12 КоАП РФ
}

# Коэффициенты сложности обнаружения (K_detection)
DETECTION_TIME_MULTIPLIER: dict[str, float] = {
    "BenignTraffic": 1.0,  # базовое значение, угрозы нет
    "Recon": 1.1,  # минуты/часы [PT 2024, экспертная оценка]
    "DDoS": 1.2,  # 91% атак < 10 мин [Cloudflare Q4 2024]
    "Spoofing": 1.4,  # анализ транзакций [ЦБ РФ 2025, экспертная оценка]
    "DoS": 1.8,  # 236 дней цикл [IBM 2025, K = 236/267 * 2.0]
    "Web-based": 1.8,  # 245 дней цикл [IBM 2025, K = 245/267 * 2.0]
    "BruteForce": 1.8,  # 246 дней цикл [IBM 2025, K = 246/267 * 2.0]
    "Mirai": 2.0,  # 267 дней цикл, эталон [IBM 2025, Barracuda 2025]
}
