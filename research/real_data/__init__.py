"""Verified real robot data ingestion, separate from synthetic experiments."""

from .loader import ForecastWindow, RolloutDataset, load_npz, load_split

__all__ = ["ForecastWindow", "RolloutDataset", "load_npz", "load_split"]
