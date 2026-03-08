class ApplicationError(Exception):
    """Базовое исключение для слоя приложения."""

    detail: str

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class ModelNotTrainedError(ApplicationError):
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Модель '{model_name}' еще не обучена.")


class DataNotLoadedError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            "Набор данных не загружен. Сначала запустите LoadDatasetInteractor"
        )


class DatasetNotFoundError(ApplicationError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Набор данных, не найденный по адресу: {path}")
