import sys

import pytest

from humanoid.utils.paths import (
    HUMANOID_DATA_DIR_ENVIRONMENT_VARIABLE,
    find_data_root,
    find_repo_root,
)


def test_find_repo_root_from_nested_directory(tmp_path):
    root = tmp_path / "repo"
    nested = root / "src" / "package"
    nested.mkdir(parents=True)
    (root / ".git").mkdir()

    assert find_repo_root(nested) == root.resolve()


def test_find_repo_root_accepts_worktree_git_file(tmp_path):
    root = tmp_path / "worktree"
    source = root / "tests" / "test_example.py"
    source.parent.mkdir(parents=True)
    source.touch()
    (root / ".git").write_text("gitdir: ../.git/worktrees/example\n")

    assert find_repo_root(source) == root.resolve()


def test_find_repo_root_raises_when_marker_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not find a repository root"):
        find_repo_root(tmp_path)


def test_default_start_uses_current_directory(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / ".git").mkdir()
    monkeypatch.chdir(nested)

    assert find_repo_root() == root.resolve()


def test_data_root_prefers_environment_override(monkeypatch, tmp_path):
    configured_root = tmp_path / "data"
    monkeypatch.setenv(HUMANOID_DATA_DIR_ENVIRONMENT_VARIABLE, str(configured_root))

    assert find_data_root(tmp_path) == configured_root.resolve()


def test_data_root_uses_virtual_environment_parent_without_repo(monkeypatch, tmp_path):
    install_root = tmp_path / "humanoid"
    module = install_root / ".venv" / "lib" / "python3.13" / "site-packages" / "module.py"
    module.parent.mkdir(parents=True)
    module.touch()
    monkeypatch.delenv(HUMANOID_DATA_DIR_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(sys, "prefix", str(install_root / ".venv"))
    monkeypatch.setattr(sys, "base_prefix", "/usr")

    assert find_data_root(module) == install_root.resolve()
