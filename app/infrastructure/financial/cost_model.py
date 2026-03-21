BASE_COSTS: dict[str, float] = {
    # BenignTraffic: потерь нет
    "BenignTraffic": 0.0,
    # DDoS: $408,000 * 92.66 = 37,805,280 ₽
    # Источник [3]: $6,000/мин * 68 мин (средняя атака 2023)
    "DDoS": 37_805_280.0,
    # DoS: $270,000 * 92.66 = 25,018,200 ₽
    # Источник [4]: $6,000/мин * 45 мин (H1 2024)
    "DoS": 25_018_200.0,
    # Mirai/Botnet: $5,130,000 * 92.66 = 475,345,800 ₽
    # Источник [5]: средние потери от botnet/ransomware атаки
    "Mirai": 475_345_800.0,
    # BruteForce: $4,810,000 * 92.66 = 445,693,600 ₽
    # Источник [1]: stolen credentials vector, среднее $4.81M
    # Примечание: 292 дня — рекордное время обнаружения среди всех векторов [1]
    "BruteForce": 445_693_600.0,
    # Web-based: $650,000 * 92.66 = 60,229,000 ₽
    # Источник [1]: $173/запись * ~3,750 записей (типичная веб-атака)
    "Web-based": 60_229_000.0,
    # Recon: $50,000 * 92.66 = 4,633,000 ₽
    # Источник: Positive Technologies — Актуальные киберугрозы 2023-2024
    # ptsecurity.com/ru-ru/research/analytics — прямой ущерб от разведки минимален
    "Recon": 4_633_000.0,
    # Spoofing: $1,200,000 * 92.66 = 111,192,000 ₽
    # Источник [2]: MITM/ARP-spoofing инциденты, Verizon DBIR 2024
    "Spoofing": 111_192_000.0,
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
