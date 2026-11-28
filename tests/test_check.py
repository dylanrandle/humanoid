import os
from unittest.mock import MagicMock, call

from humanoid import check


def test_check_runs_complete_suite_from_repository_root(monkeypatch):
    project_root = check.find_repo_root(__file__)
    run = MagicMock()
    monkeypatch.setattr(check.subprocess, "run", run)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setenv("HUMANOID_CHECK_TEST", "preserved")

    check.main()

    assert run.call_args_list == [
        call(
            command,
            cwd=project_root,
            check=True,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "LP_NUM_THREADS": "1",
            },
        )
        for command in check.CHECK_COMMANDS
    ]
