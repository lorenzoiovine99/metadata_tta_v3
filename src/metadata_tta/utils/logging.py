from __future__ import annotations

import logging
import sys
from pathlib import Path


LOGGER_NAME = "metadata_tta"


def configure_logging(
    *,
    run_directory: str | Path | None,
    level: str = "INFO",
    console: bool = True,
    save_to_file: bool = True,
) -> logging.Logger:

    logger = logging.getLogger(
        LOGGER_NAME
    )

    numeric_level = getattr(
        logging,
        str(level).upper(),
        logging.INFO,
    )

    logger.setLevel(
        numeric_level
    )

    logger.propagate = False

    # Important for batch execution:
    # do not accumulate handlers across runs.
    for handler in list(
        logger.handlers
    ):
        handler.close()
        logger.removeHandler(
            handler
        )

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:

        console_handler = (
            logging.StreamHandler(
                sys.stdout
            )
        )

        console_handler.setLevel(
            numeric_level
        )

        console_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            console_handler
        )

    if save_to_file:

        if run_directory is None:
            raise ValueError(
                "save_to_file=True requires "
                "run_directory."
            )

        run_directory = Path(
            run_directory
        )

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.FileHandler(
            run_directory / "run.log",
            mode="a",
            encoding="utf-8",
        )

        file_handler.setLevel(
            numeric_level
        )

        file_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

    return logger


def get_logger() -> logging.Logger:

    return logging.getLogger(
        LOGGER_NAME
    )