from __future__ import annotations

import logging


def get_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return logging.getLogger("ams")
