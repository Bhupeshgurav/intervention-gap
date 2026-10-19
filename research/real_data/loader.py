"""Small, explicit loader for the real Transferable Dynamics Learning splits.

Only NumPy is required. Arrays keep rollout boundaries, physical units and all
five original field names. No automatic resampling, normalization or splitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = (
    "measured_angles", "measured_velocities", "measured_torques",
    "constrained_torques", "desired_torques",
)
UNITS = {
    "measured_angles": "rad",
    "measured_velocities": "rad/s",
    "measured_torques": "N*m",
    "constrained_torques": "N*m",
    "desired_torques": "N*m",
}
SPLITS = ("training", "validation", "test_iid", "test_transfer_1", "test_transfer_2", "test_transfer_3")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "transferable_dynamics"
SAMPLE_INTERVAL_SECONDS = 0.001  # Original paper, not the README's erroneous '1 Hz'.


@dataclass(frozen=True)
class ForecastWindow:
    """Conditional replay window, not a prospective control experiment.

    History ends at t. Candidate/logged desired commands cover t..t+h-1.
    Targets cover t+1..t+h. Future measured and constrained torques are excluded.
    This follows the original loader's action-to-next-observation convention;
    it does not establish exact hardware sensor/actuator synchronization.
    """

    rollout_id: str
    anchor_index: int
    history: dict[str, np.ndarray]
    future_desired_commands: np.ndarray
    target_angles: np.ndarray
    target_velocities: np.ndarray


@dataclass(frozen=True)
class RolloutDataset:
    path: Path
    arrays: dict[str, np.ndarray]
    sample_interval_seconds: float = SAMPLE_INTERVAL_SECONDS

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.arrays["measured_angles"].shape

    @property
    def rollout_ids(self) -> tuple[str, ...]:
        return tuple(f"{self.path.stem}:{i}" for i in range(self.shape[0]))

    @property
    def time_seconds(self) -> np.ndarray:
        """Relative sample times; no original wall-clock timestamps are supplied."""
        return np.arange(self.shape[1]) * self.sample_interval_seconds

    def forecast_window(self, rollout: int, anchor: int, history: int, horizon: int) -> ForecastWindow:
        for label, value in (("rollout", rollout), ("anchor", anchor), ("history", history), ("horizon", horizon)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{label} must be an integer")
        if history < 1 or horizon < 1:
            raise ValueError("history and horizon must be positive")
        if not 0 <= rollout < self.shape[0]:
            raise IndexError("Rollout index outside this split")
        if anchor < history - 1 or anchor + horizon >= self.shape[1]:
            raise IndexError("Window would cross a rollout boundary")
        past = {name: value[rollout, anchor-history+1:anchor+1] for name, value in self.arrays.items()}
        return ForecastWindow(
            rollout_id=self.rollout_ids[rollout], anchor_index=anchor, history=past,
            future_desired_commands=self.arrays["desired_torques"][rollout, anchor:anchor+horizon],
            target_angles=self.arrays["measured_angles"][rollout, anchor+1:anchor+horizon+1],
            target_velocities=self.arrays["measured_velocities"][rollout, anchor+1:anchor+horizon+1],
        )


def load_split(family: str, split: str, directory: str | Path = DATA_DIR, *, verify_checksum: bool = True) -> RolloutDataset:
    if family not in ("Sines", "GPs") or split not in SPLITS:
        raise ValueError("Choose a real Sines/GPs family and one official split")
    return load_npz(Path(directory) / f"{family}_{split}.npz", verify_checksum=verify_checksum)


def load_npz(path: str | Path, *, verify_checksum: bool = True) -> RolloutDataset:
    path = Path(path)
    if path.name.startswith("Sim_"):
        raise ValueError("This loader deliberately refuses simulated data")
    if verify_checksum:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        expected = next((item for item in manifest["files"] if item["filename"] == path.name), None)
        if expected is None or not expected.get("is_real_measurement"):
            raise ValueError("File absent from the verified real-data manifest")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("Dataset SHA-256 does not match manifest")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(FIELDS):
            raise ValueError(f"Unexpected fields: {archive.files}")
        arrays = {name: archive[name] for name in FIELDS}
    shape = arrays[FIELDS[0]].shape
    if len(shape) != 3 or shape[0] < 1 or shape[1] < 2 or shape[2] != 3:
        raise ValueError(f"Expected (rollouts, time, 3), got {shape}")
    for name, array in arrays.items():
        if array.shape != shape or array.dtype.kind not in "fiu":
            raise ValueError(f"Invalid numeric shape/dtype for {name}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite data in {name}; missingness must be handled explicitly")
        array.flags.writeable = False
    return RolloutDataset(path=path, arrays=arrays)
