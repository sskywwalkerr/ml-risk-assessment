class InfrastructureError(Exception):
    """Базовое исключение для доменного слоя."""

    detail: str

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class ModelLoadError(InfrastructureError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Не удалось загрузить модель: '{name}'")


class DataReadError(InfrastructureError):
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Не удалось считать данные из'{path}': {reason}")
