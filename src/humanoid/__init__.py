import os

from humanoid.logger import setup_logging

setup_logging(os.environ.get("LOGLEVEL", "INFO"))
