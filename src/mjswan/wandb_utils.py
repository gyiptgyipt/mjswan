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


def fetch_pt_onnx_from_wandb_run(
    run_path: str,
    task_id: str,
) -> list[tuple[str, onnx.ModelProto]]:
    """Download all ``model_*.pt`` checkpoints from a W&B run and convert each to ONNX.

    Checkpoints are sorted by training step before conversion, so the returned
    list is ordered from earliest to latest (e.g. ``model_0``, ``model_50``, …).

    The mjlab environment is constructed once and reused across all checkpoints
    to avoid repeated startup cost.  Requires ``mjlab`` and ``torch`` to be
    installed.

    Args:
        run_path: W&B run path in the format ``"entity/project/run_id"``.
        task_id: mjlab task identifier (e.g. ``"go2_flat"``). Used to
            reconstruct the environment and runner required for ONNX export.

    Returns:
        List of ``(policy_name, onnx_model)`` tuples sorted by training step,
        where policy_name is the filename without extension (e.g. ``"model_50"``).

    Raises:
        ImportError: If ``mjlab`` or ``torch`` are not installed.
        ValueError: If no ``model_*.pt`` files are found in the run.
    """
    import re
    from dataclasses import asdict

    import wandb

    try:
        import mjlab.tasks  # noqa: F401 — populates the task registry
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    except ImportError as e:
        raise ImportError(
            "mjlab and torch are required for only_latest=False. "
            "Install them with: pip install mjlab torch"
        ) from e

    api = wandb.Api()
    run = api.run(run_path)

    pt_files = [f for f in run.files() if re.match(r"^model_\d+\.pt$", f.name)]

    if not pt_files:
        raise ValueError(f"No model_*.pt files found in W&B run: {run_path}")

    # Sort by training step number (model_0.pt < model_50.pt < model_100.pt, …)
    pt_files.sort(key=lambda f: int(re.search(r"\d+", f.name).group()))  # type: ignore[union-attr]

    results: list[tuple[str, onnx.ModelProto]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Build mjlab environment once — reused across all checkpoints.
        env_cfg = load_env_cfg(task_id, play=True)
        env_cfg.scene.num_envs = 1
        agent_cfg = load_rl_cfg(task_id)

        env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        try:
            runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
            runner = runner_cls(env, asdict(agent_cfg), device="cpu")

            for wandb_file in pt_files:
                wandb_file.download(root=tmp_dir, replace=True)
                pt_path = tmp_path / wandb_file.name
                name = pt_path.stem  # e.g. "model_0", "model_50"
                onnx_filename = f"{name}.onnx"

                runner.load(
                    str(pt_path),
                    load_cfg={"actor": True},
                    strict=True,
                    map_location="cpu",
                )
                runner.export_policy_to_onnx(tmp_dir, onnx_filename)

                model = onnx.load(str(tmp_path / onnx_filename))
                results.append((name, model))
        finally:
            env.close()

    return results


__all__ = ["fetch_onnx_from_wandb_run", "fetch_pt_onnx_from_wandb_run"]
