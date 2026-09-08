import logging
import time
from multiprocessing import get_context
from multiprocessing.queues import Queue

from humanoid.config.process import PROCESS_START_METHOD
from humanoid.logger import setup_queue_logging
from humanoid.orchestrator.monitor.application_log import ApplicationLogMonitor

TEST_RETAINED_ENTRIES = 2
TEST_MAX_FILE_BYTES = 256


def _write_child_log(log_queue: Queue[logging.LogRecord]) -> None:
    setup_queue_logging(log_queue)
    logging.getLogger("humanoid.child").warning("Child node started")


def test_application_log_monitor_returns_incremental_bounded_snapshots(tmp_path):
    log_path = tmp_path / "humanoid.log"
    log_path.write_text("prior one\nprior two\n")
    monitor = ApplicationLogMonitor(log_path, retained_entries=TEST_RETAINED_ENTRIES)
    logger = logging.getLogger("humanoid.test")
    access_logger = logging.getLogger("werkzeug")
    previous_access_level = access_logger.level
    access_logger.setLevel(logging.INFO)

    try:
        initial = monitor.snapshot()
        assert [entry.message for entry in initial.entries] == ["prior one", "prior two"]

        logger.info("current one")
        access_logger.info("GET /api/application-logs")
        incremental = monitor.snapshot(after=initial.cursor)
        assert len(incremental.entries) == 1
        assert "current one" in incremental.entries[0].message
        assert "GET /api/application-logs" not in log_path.read_text()
        assert incremental.reset is False

        logger.info("current two")
        logger.info("current three")
        reset = monitor.snapshot(after=initial.cursor)
        assert reset.reset is True
        assert len(reset.entries) == TEST_RETAINED_ENTRIES
        assert reset.capacity == TEST_RETAINED_ENTRIES
    finally:
        access_logger.setLevel(previous_access_level)
        monitor.close()


def test_application_log_monitor_serializes_child_logs_and_rotates_file(tmp_path):
    log_path = tmp_path / "humanoid.log"
    log_path.write_text("old output" * 100)
    monitor = ApplicationLogMonitor(
        log_path,
        retained_entries=20,
        max_file_bytes=TEST_MAX_FILE_BYTES,
        backup_count=1,
        initial_tail_bytes=128,
    )

    try:
        assert log_path.stat().st_size <= TEST_MAX_FILE_BYTES
        child = get_context(PROCESS_START_METHOD).Process(
            target=_write_child_log,
            args=(monitor.queue,),
        )
        child.start()
        child.join(timeout=5)
        assert child.exitcode == 0

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if any("Child node started" in entry.message for entry in monitor.snapshot().entries):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Child log did not reach the parent listener")

        logging.getLogger("humanoid.test").warning("x" * 120)
        assert log_path.stat().st_size <= TEST_MAX_FILE_BYTES
        assert log_path.with_suffix(".log.1").exists()
    finally:
        monitor.close()
