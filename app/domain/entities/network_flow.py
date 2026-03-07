from dataclasses import dataclass, field


@dataclass(frozen=True)
class NetworkFlow:
    """Один поток сетевого трафика из датасета."""

    flow_duration: float
    flow_bytes_per_s: float
    flow_packets_per_s: float
    total_fwd_packets: float
    total_bwd_packets: float
    fwd_packet_length_mean: float
    bwd_packet_length_mean: float
    flow_iat_mean: float
    flow_iat_std: float
    label: str
    extra_features: dict = field(default_factory=dict)
