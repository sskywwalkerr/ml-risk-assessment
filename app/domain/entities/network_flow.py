from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class NetworkFlow:
    """Один поток сетевого трафика из датасета CICIoT2023."""

    # Временны́е характеристики потока
    flow_duration: float
    header_length: float
    protocol_type: float
    duration: float

    # Скорости передачи
    rate: float
    srate: float
    drate: float

    # TCP флаги
    fin_flag_number: float
    syn_flag_number: float
    rst_flag_number: float
    psh_flag_number: float
    ack_flag_number: float
    ece_flag_number: float
    cwr_flag_number: float

    # Счётчики флагов
    ack_count: float
    syn_count: float
    fin_count: float
    urg_count: float
    rst_count: float

    # Протоколы приложений (бинарные признаки)
    http: float
    https: float
    dns: float
    ssh: float
    tcp: float
    udp: float
    icmp: float

    # Статистики пакетов
    tot_sum: float
    pkt_min: float
    pkt_max: float
    pkt_avg: float
    pkt_std: float
    tot_size: float
    iat: float
    number: float
    magnitude: float
    radius: float
    covariance: float
    variance: float
    weight: float

    label: str
    extra: dict = field(default_factory=dict)
