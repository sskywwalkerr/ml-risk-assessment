import numpy as np
import pandas as pd

from app.application.interfaces.feature_engineer import IFeatureEngineer

# Признаки из датасета
_RATE_COLS = ("rate", "srate", "drate")
_FLAG_COLS = (
    "syn_flag_number",
    "ack_flag_number",
    "fin_flag_number",
    "rst_flag_number",
    "psh_flag_number",
)
_COUNT_COLS = ("syn_count", "ack_count", "fin_count", "rst_count", "urg_count")
_PROTO_COLS = ("tcp", "udp", "icmp", "http", "https")


class FeatureEngineer(IFeatureEngineer):
    """Создаёт производные признаки из сырого трафика.
    Группы признаков:
      1. Traffic ratios    - соотношения скоростей трафика
      2. Flag features     - агрегаты TCP-флагов
      3. Packet statistics - статистики размеров пакетов
      4. Anomaly scores    - Z-оценки аномальности потока
      5. Risk indicators   - составные индикаторы риска атаки
    """

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._traffic_ratios(df)
        df = self._flag_features(df)
        df = self._packet_statistics(df)
        df = self._anomaly_scores(df)
        df = self._risk_indicators(df)
        return df

    def _traffic_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Соотношения скоростей трафика."""

        rate = self._col(df, "rate")
        srate = self._col(df, "srate")
        drate = self._col(df, "drate")
        dur = self._col(df, "flow_duration")

        if rate is not None and drate is not None:
            # Асимметрия трафика: отношение входящего к исходящему
            df["src_dst_rate_ratio"] = rate / (drate + 1e-8)

        if srate is not None and drate is not None:
            # Разница скоростей - индикатор однонаправленных атак
            df["rate_asymmetry"] = (srate - drate).abs()

        if rate is not None and dur is not None:
            # Интенсивность потока = скорость * длительность
            df["flow_intensity"] = rate * dur.clip(lower=0)

        return df

    def _flag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Агрегаты TCP-флагов."""

        flag_series: dict[str, pd.Series] = {
            column: series
            for column in _FLAG_COLS
            if (series := self._col(df, column)) is not None
        }
        count_series: dict[str, pd.Series] = {
            column: series
            for column in _COUNT_COLS
            if (series := self._col(df, column)) is not None
        }

        if flag_series:
            # Суммарное число активных флагов - аномально высокое значение = атака
            flag_values = list(flag_series.values())
            total_flags = flag_values[0].copy()
            for flag in flag_values[1:]:
                total_flags = total_flags + flag
            df["total_flags"] = total_flags

        syn_flag = flag_series.get("syn_flag_number")
        ack_flag = flag_series.get("ack_flag_number")
        if syn_flag is not None and ack_flag is not None:
            # SYN без ACK - признак SYN-flood
            # syn = запрос на установление соединения, ack = подтверждение получения данных
            df["syn_without_ack"] = (syn_flag - ack_flag).clip(
                lower=0
            )  #  гарантирует, что если результат ушел в минус, он превратится в ноль.

        if count_series:
            # Суммарное число пакетов по всем флагам.
            count_values = list(count_series.values())
            total_flag_count = count_values[0].copy()
            for count in count_values[1:]:
                total_flag_count = total_flag_count + count
            df["total_flag_count"] = total_flag_count

        syn_count = count_series.get("syn_count")
        rst_count = count_series.get("rst_count")
        if syn_count is not None and rst_count is not None:
            # Высокое соотношение RST/SYN - признак отказа в соединении
            # RST - мгновенное принудительное прерывание соединения
            df["rst_syn_ratio"] = rst_count / (syn_count + 1e-8)

        return df

    def _packet_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Производные статистики из размеров пакетов."""

        pkt_min = self._col(df, "min")
        pkt_max = self._col(df, "max")
        pkt_avg = self._col(df, "avg")
        pkt_std = self._col(df, "std")

        if pkt_max is not None and pkt_min is not None:
            # Разброс размеров пакетов - у DDoS обычно однородные пакеты
            df["pkt_size_range"] = pkt_max - pkt_min

        if pkt_std is not None and pkt_avg is not None:
            # Коэффициент вариации(мера разброса) - относительный разброс
            df["pkt_cv"] = pkt_std / (pkt_avg + 1e-8)
        magnitude = self._col(df, "magnitude")  # величина
        radius = self._col(df, "radius")  # радиус
        covariance = self._col(
            df, "covariance"
        )  # мера линейной зависимости двух случайных величин

        if magnitude is not None and radius is not None:
            # Произведение магнитуды и радиуса - мера "объёма" трафика
            df["traffic_volume"] = magnitude * radius
        if covariance is not None:
            # Логарифм ковариации - сжимает большой диапазон значений
            df["covariance_log"] = np.log1p(covariance.clip(lower=0))
        return df

    def _anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-оценки аномальности для ключевых признаков.
        Отклонения от среднего значения в выборке в единицах стандартного отклонения
        """
        for col in ("rate", "header_length", "flow_duration", "iat"):
            s = self._col(df, col)
            if s is None:
                continue
            mean, std = s.mean(), s.std()
            if pd.isna(mean) or pd.isna(std) or std == 0:
                df[f"{col}_zcore"] = 0.0
            else:
                df[f"{col}_zcore"] = ((s - mean) / (std + 1e-10)).abs()
        return df

    def _risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Составные индикаторы риска специфичные для IoT-атак."""

        rate = self._col(df, "rate")  # коэффициент
        weight = self._col(df, "weight")  # вес для расчета средневзвешенных показателей
        number = self._col(df, "number")  # номер
        iat = self._col(df, "iat")  # время между событиями

        if rate is not None and weight is not None:
            # Взвешенная скорость — признак аномально нагруженных потоков
            df["weighted_rate"] = rate * weight
        if number is not None and iat is not None:
            # Плотность пакетов во времени - высокая = DoS/DDoS
            df["packet_density"] = number / (iat + 1e-8)

        # Флаги протоколов часто используемых в атаках
        tcp = self._col(df, "tcp")
        udp = self._col(df, "udp")
        icmp = self._col(df, "icmp")

        if tcp is not None and udp is not None and icmp is not None:
            # Энтропия протоколов - разнообразие = нормальный трафик
            total = tcp + udp + icmp + 1e-8
            df["protocol_entropy"] = -(
                (tcp / total) * np.log2(tcp / total + 1e-8)
                + (udp / total) * np.log2(udp / total + 1e-8)
                + (icmp / total) * np.log2(icmp / total + 1e-8)
            )
        return df

    @staticmethod
    def _col(df: pd.DataFrame, col: str) -> pd.Series | None:
        """Безопасно возвращает числовую колонку или None."""
        if col not in df.columns:
            return None
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
