import logging

from acn.config.settings import Settings

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level,
        format=LOG_FORMAT,
    )
