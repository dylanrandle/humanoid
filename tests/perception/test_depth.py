from pathlib import Path

import numpy as np
import pytest

from humanoid.perception.depth import DepthCompleter, _metres, _prepare


def test_depth_units_invalid_samples_and_rgb_order():
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[:, :, 0] = 255
    depth = np.full((480, 640), 2500, np.uint16)
    depth[:, :320] = 0
    tensor = _prepare(rgb, depth, 0.0001)
    assert tensor.shape == (1, 4, 518, 686)
    np.testing.assert_allclose(
        tensor[0, :3, 0, 0], [-0.485 / 0.229, -0.456 / 0.224, 0.594 / 0.225], rtol=0.001
    )
    assert tensor[0, 3, 0, 0] == 0
    assert tensor[0, 3, 0, -1] == 4  # noqa: PLR2004
    output = _metres(tensor[:, 3])
    assert output.shape == depth.shape
    assert output.dtype == np.float32
    assert np.isnan(output[:, 0]).all()
    np.testing.assert_equal(output[:, -1], 0.25)


def test_git_lfs_pointer_diagnosed_before_gpu_load(tmp_path: Path):
    (tmp_path / "d435.plan").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="git lfs pull"):
        DepthCompleter("d435", tmp_path)
