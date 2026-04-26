from app.infrastructure.config import Config
from app.infrastructure.logger import setup_logging
from app.main.pipeline import Pipeline


def main() -> None:
    config = Config.from_yaml("app/main/config.yaml")
    setup_logging(config.app)
    Pipeline(config).run()


if __name__ == "__main__":
    main()
