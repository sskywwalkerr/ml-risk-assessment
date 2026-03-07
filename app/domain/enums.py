from enum import Enum, StrEnum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackCategory(StrEnum):
    BENIGN = "benign"
    DDOS = "DDoS"
    DOS = "DoS"
    RECON = "Recon"
    WEB_BASED = "Web-based"
    BRUTE_FORCE = "BruteForce"
    SPOOFING = "Spoofing"
    MIRAI = "Mirai"
