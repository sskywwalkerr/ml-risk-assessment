import logging.config
from typing import Any

from app.infrastructure.config import AppConfig


def setup_logging(config: AppConfig) -> None:
    formatter = "standard" if config.debug else "json"

    log_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                "datefmt": "%H:%M:%S",
            },
            "json": {
                "()": "pythonjsonlogger.json.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "json_ensure_ascii": False,
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": formatter,
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": True,
            },
            "app": {
                "handlers": ["default"],
                "level": "DEBUG" if config.debug else "INFO",
                "propagate": False,
            },
            # Загрузка данных
            "app.infrastructure.data": {
                "handlers": ["default"],
                "level": "DEBUG" if config.debug else "INFO",
                "propagate": False,
            },
            # Обучение моделей
            "app.infrastructure.models": {
                "handlers": ["default"],
                "level": "DEBUG" if config.debug else "INFO",
                "propagate": False,
            },
            # Финансовый модуль
            "app.infrastructure.financial": {
                "handlers": ["default"],
                "level": "DEBUG" if config.debug else "INFO",
                "propagate": False,
            },
            # Шумные библиотеки
            "lightgbm": {
                "handlers": ["default"],
                "level": "WARNING",
                "propagate": False,
            },
            "xgboost": {
                "handlers": ["default"],
                "level": "WARNING",
                "propagate": False,
            },
            "sklearn": {
                "handlers": ["default"],
                "level": "WARNING",
                "propagate": False,
            },
            "matplotlib": {
                "handlers": ["default"],
                "level": "WARNING",
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(log_config)
