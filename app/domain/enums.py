from enum import StrEnum


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AttackCategory(StrEnum):
    BENIGN = "BenignTraffic"
    DDOS = "DDoS"
    DOS = "DoS"
    RECON = "Recon"
    WEB_BASED = "Web-based"
    BRUTE_FORCE = "BruteForce"
    SPOOFING = "Spoofing"
    MIRAI = "Mirai"
