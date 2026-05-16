import functools
import sys
from abc import ABC, abstractmethod

from humanoid.logger import get_logger
from humanoid.loop import loop_at_rate

logger = get_logger(__name__)


class Node(ABC):
    rate_hz: float

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "__init__" not in cls.__dict__:
            return
        original = cls.__init__

        @functools.wraps(original)
        def logged_init(self, *args, **kw):
            logger.info(f"Initializing {type(self).__name__}")
            original(self, *args, **kw)
            logger.info(f"{type(self).__name__} initialized")

        cls.__init__ = logged_init  # ty:ignore[invalid-assignment]

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def on_close(self) -> None: ...

    @abstractmethod
    def setup(self) -> None: ...

    @classmethod
    def main(cls, *args, **kwargs) -> None:
        cls(*args, **kwargs).run()

    def stop_condition(self) -> bool:
        return False

    def run(self) -> None:
        logger.info(f"Starting {type(self).__name__} at {self.rate_hz} Hz...")
        try:
            self.setup()
            loop_at_rate(self.step, rate_hz=self.rate_hz, stop_condition=self.stop_condition)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            sys.exit(1)
        finally:
            self.close()

    def close(self) -> None:
        logger.info(f"Closing {type(self).__name__}...")
        self.on_close()
        logger.info(f"{type(self).__name__} closed")
