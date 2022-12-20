"""
This file handles the logger used throughout the use of the code
"""

import logging
from src.utils.config import config

# setup logger
level = config.logging.level.upper()

if level == "DEBUG":
    logging.basicConfig(level=logging.DEBUG)
elif level == "INFO":
    logging.basicConfig(level=logging.INFO)
elif level == "WARNING":
    logging.basicConfig(level=logging.WARNING)
elif level == "ERROR":
    logging.basicConfig(level=logging.ERROR)
elif level == "CRITICAL":
    logging.basicConfig(level=logging.CRITICAL)
else:
    raise ValueError(
        "Not a defined logging level, choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', and 'CRITICAL'."
    )
log = logging.getLogger(config.logging.name)
handler = logging.FileHandler(config.logging.filename)
