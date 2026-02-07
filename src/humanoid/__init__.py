import os

from humanoid.logging import setup_logging

setup_logging(os.environ.get("LOGLEVEL", "INFO"))
