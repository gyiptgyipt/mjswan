"""Utilities for loading ONNX policies from Weights & Biases."""

from __future__ import annotations

import tempfile
from pathlib import Path

import onnx


def fetch_onnx_from_wandb_run(run_path: str) -> list[tuple[str, onnx.ModelProto]]:
    """Download all ONNX policy files from a W&B run.

    Fetches every ``.onnx`` file attached to the run and loads each one into
    memory as an :class:`onnx.ModelProto`.  The policy name for each file is
    the filename with its extension removed (e.g. ``"2026-02-25_04-30-08.onnx"``
    becomes ``"2026-02-25_04-30-08"``).

    Args:
        run_path: W&B run path in the format ``"entity/project/run_id"``.

    Returns:
        List of ``(policy_name, onnx_model)`` tuples, one per ``.onnx`` file
        found in the run.

    Raises:
        ValueError: If no ``.onnx`` files are found in the run.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    onnx_files = [f for f in run.files() if f.name.endswith(".onnx")]

    if not onnx_files:
        raise ValueError(f"No .onnx files found in W&B run: {run_path}")

    results: list[tuple[str, onnx.ModelProto]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for wandb_file in onnx_files:
            wandb_file.download(root=tmp_dir, replace=True)
            local_path = Path(tmp_dir) / wandb_file.name
            name = local_path.stem
            model = onnx.load(str(local_path))
            results.append((name, model))

    return results


__all__ = ["fetch_onnx_from_wandb_run"]
