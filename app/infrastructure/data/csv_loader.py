import logging
from pathlib import Path

import pandas as pd

from app.application.interfaces.data_loader import IDataLoader, LoadDataRequest
from app.infrastructure.exceptions import DataReadError

logger = logging.getLogger(__name__)

LABEL_MAP: dict[str, str] = {
    # Benign
    "BenignTraffic": "BenignTraffic",
    # DDoS
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    # DoS
    "DoS-UDP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # Recon
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon",
    "Recon-PingSweep": "Recon",
    "VulnerabilityScan": "Recon",
    # Web-based
    "SqlInjection": "Web-based",
    "CommandInjection": "Web-based",
    "Backdoor_Malware": "Web-based",
    "Uploading_Attack": "Web-based",
    "XSS": "Web-based",
    "BrowserHijacking": "Web-based",
    # BruteForce
    "BruteForce-Web": "BruteForce",
    "BruteForce-XSS": "BruteForce",
    "DictionaryBruteForce": "BruteForce",
    # Spoofing
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    # Mirai
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
}


class CSVDataLoader(IDataLoader):
    def __init__(self, data_path: str) -> None:
        self._path = Path(data_path)

    def load(self, request: LoadDataRequest) -> pd.DataFrame:
        csv_files = sorted(self._path.glob("*.csv"))
        if not csv_files:
            raise DataReadError(str(self._path), "CSV-файлы не найдены.")

        frames = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, skipinitialspace=True, low_memory=False)
                frames.append(df)
                logger.info("Загрузка: %s, строк: %d", f.name, len(df))
            except Exception as e:
                raise DataReadError(str(f), str(e)) from e

        data = pd.concat(frames, ignore_index=True)
        data.columns = [c.strip().lower() for c in data.columns]

        if "label" not in data.columns:
            raise DataReadError(
                str(self._path), "Столбец 'label' не найден в наборе данных"
            )

        data["label"] = (
            data["label"].astype(str).str.strip().map(LABEL_MAP).fillna("BenignTraffic")
        )

        # Стратификация выборки
        if request.sample_size and request.sample_size < len(data):
            data = (
                data.groupby("label", group_keys=False)
                .apply(
                    lambda x: x.sample(
                        min(len(x), int(request.sample_size * len(x) / len(data))),
                        random_state=42,
                    )
                )
                .reset_index(drop=True)
            )

        logger.info("Всего: %d строк | %d столбцов", len(data), len(data.columns))
        return data
