from __future__ import annotations

import subprocess

import pytest

from ragbench.evaluation import runner as runner_module
from ragbench.evaluation.runner import _environment_metadata


def test_environment_metadata_has_every_expected_field() -> None:
    metadata = _environment_metadata()
    for key in (
        "os",
        "os_release",
        "machine",
        "cpu",
        "logical_cores",
        "total_ram_bytes",
        "gpu",
        "torch_device",
        "numpy_blas_backend",
        "omp_num_threads",
        "mkl_num_threads",
        "chat_model",
        "embed_model",
    ):
        assert key in metadata


def test_environment_metadata_never_raises_even_when_every_optional_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a missing optional dependency or unavailable command must
    degrade a field to None, not break the whole benchmark run just to
    report metadata about it."""

    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no /proc/cpuinfo")))

    metadata = _environment_metadata()  # must not raise
    assert metadata["gpu"] is None


def test_gpu_info_returns_none_when_nvidia_smi_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    assert runner_module._gpu_info() is None


def test_gpu_info_parses_nvidia_smi_csv_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="NVIDIA A100, 40960 MiB, 535.104.05\n")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    assert runner_module._gpu_info() == {
        "name": "NVIDIA A100",
        "memory_total": "40960 MiB",
        "driver_version": "535.104.05",
    }
