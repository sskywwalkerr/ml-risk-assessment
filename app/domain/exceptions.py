class DomainError(Exception):
    """Базовое исключение для доменного слоя."""

    detail: str

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class UnknownAttackError(DomainError):
    def __init__(self, attack_type: str) -> None:
        super().__init__(f"Неизвестный тип атаки: {attack_type}")


class InvalidFlowError(DomainError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Недопустимый сетевой поток: {reason}")
