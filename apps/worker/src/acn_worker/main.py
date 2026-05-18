import logging

from acn.config.logging import configure_logging
from acn.config.settings import Settings, get_settings


def run(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings)
    logger = logging.getLogger("acn.worker")
    logger.info("worker.started", extra={"environment": resolved_settings.env})


def main() -> None:
    run()


if __name__ == "__main__":
    main()
