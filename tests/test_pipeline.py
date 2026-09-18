from __future__ import annotations

import pytest

from src.ee_utils import add_eetc_to_path
from src.pipeline import CropTypePipeline


def test_add_eetc_to_path_missing_raises(tmp_config):
    tmp_config.eetc_path = tmp_config.project_root / "no_such_eetc"
    with pytest.raises(FileNotFoundError, match="eetc repo not found"):
        add_eetc_to_path(tmp_config)


def test_add_eetc_to_path_inserts_sys_path(tmp_config, monkeypatch):
    import sys

    eetc = tmp_config.project_root / "eetc"
    (eetc / "gee_tools").mkdir(parents=True)
    tmp_config.eetc_path = eetc
    monkeypatch.setattr(sys, "path", list(sys.path))
    returned = add_eetc_to_path(tmp_config)
    assert returned == eetc
    assert str(eetc) in sys.path
    assert str(eetc / "gee_tools") in sys.path
    # Second call should not duplicate entries.
    add_eetc_to_path(tmp_config)
    assert sys.path.count(str(eetc)) == 1


def test_pipeline_init_creates_dirs_and_timings(tmp_config):
    pipe = CropTypePipeline(tmp_config)
    assert tmp_config.outputs_dir.is_dir()
    pipe.print_timings()
    with pipe._timed("toy"):
        pass
    assert "toy" in pipe.timings
    assert pipe.timings["toy"] >= 0
