"""Source-trained snapshot forecasting on official real-robot rollout splits.

Only inputs at the anchor are used. No future torque/command is supplied.
This is forecasting under the recorded controllers, not intervention evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np

from research.models import fit_predictor
from research.real_data.loader import load_split, SPLITS


HORIZONS = np.array([10, 50, 100])
FEATURES = {
    "state": ("measured_angles", "measured_velocities"),
    "desired": ("measured_angles", "measured_velocities", "desired_torques"),
    "execution": ("measured_angles", "measured_velocities", "desired_torques", "constrained_torques", "measured_torques"),
}


def make_windows(dataset, stride=50):
    arrays = dataset.arrays
    anchors = np.arange(100, dataset.shape[1] - int(HORIZONS.max()), stride)
    q, v = arrays["measured_angles"], arrays["measured_velocities"]
    current = q[:, anchors, :].reshape(-1, 3)
    velocity = v[:, anchors, :].reshape(-1, 3)
    target = np.stack([q[:, anchors + h, :] for h in HORIZONS], axis=2).reshape(-1, len(HORIZONS), 3)
    cv = current[:, None, :] + velocity[:, None, :] * (HORIZONS * dataset.sample_interval_seconds)[None, :, None]
    features = {label: np.concatenate([arrays[name][:, anchors, :] for name in names], axis=-1).reshape(len(current), -1) for label, names in FEATURES.items()}
    return {"features": features, "target": target, "cv": cv, "current": current, "residual": (target - cv).reshape(len(current), -1), "rollout": np.repeat(np.arange(dataset.shape[0]), len(anchors)), "anchor": np.tile(anchors, dataset.shape[0])}


def score(prediction, data, *, family, split, name, seed):
    rows = []
    for r in np.unique(data["rollout"]):
        mask = data["rollout"] == r
        errors = prediction[mask] - data["target"][mask]
        for i, h in enumerate(HORIZONS):
            rows.append({"family": family, "split": split, "model": name, "seed": seed, "rollout": int(r), "horizon_ms": int(h), "mse_rad2": float(np.mean(errors[:, i] ** 2)), "rmse_rad": float(np.sqrt(np.mean(errors[:, i] ** 2))), "samples": int(mask.sum())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/real_robot_v1")
    parser.add_argument("--families", nargs="+", default=["Sines", "GPs"], choices=["Sines", "GPs"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--stride", type=int, default=50)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError("Use a fresh output directory; do not overwrite a frozen run")
    config = {"task": "Current-input finite-horizon joint-position forecasting; no future actions or measurements", "families": args.families, "seeds": args.seeds, "epochs": args.epochs, "stride": args.stride, "horizon_samples": HORIZONS.tolist(), "sample_interval_seconds": .001, "features": FEATURES, "training": "Official source training split; source validation chooses MLP epoch; fixed ridge alpha=1; no test tuning", "target": "Position residual beyond constant-velocity prediction; target scaling fitted only on training", "models": ["persistence", "constant_velocity", "ridge_state", "ridge_desired", "ridge_execution", "mlp_state", "mlp_desired", "mlp_execution"], "network": {"hidden_layers": [64,64], "activation": "Tanh", "optimizer": "AdamW", "lr": .002, "weight_decay": .0001, "batch_size": 256}, "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path("research/models.py"), Path("research/real_data/loader.py"), Path("data/transferable_dynamics/manifest.json")]}}
    config_text = json.dumps(config, indent=2) + "\n"
    (out / "config.json").write_text(config_text)
    (out / "config.sha256").write_text(hashlib.sha256(config_text.encode()).hexdigest() + "\n")
    records, training = [], []
    for family in args.families:
        start = time.perf_counter()
        train = make_windows(load_split(family, "training"), args.stride)
        val = make_windows(load_split(family, "validation"), args.stride)
        models = []
        for kind in ("ridge", "mlp"):
            for feature in FEATURES:
                for seed in (args.seeds if kind == "mlp" else [0]):
                    name = f"{kind}_{feature}"
                    print(f"FIT {family} {name} seed={seed}", flush=True)
                    model = fit_predictor(kind, train["features"][feature], train["residual"], seed=seed, epochs=args.epochs, validation=(val["features"][feature], val["residual"]))
                    checkpoint = out / "checkpoints" / f"{family}_{name}_{seed}"
                    checkpoint.parent.mkdir(exist_ok=True)
                    if kind == "mlp":
                        model.save(checkpoint.with_suffix(".pt"))
                    else:
                        joblib.dump(model, checkpoint.with_suffix(".joblib"))
                    training.append({"family": family, "model": name, "seed": seed, "fit_seconds": model.fit_seconds, "best_epoch": getattr(model, "best_epoch", None), "parameter_count": getattr(model, "parameter_count", None), "training_windows": len(train["residual"]), "training_rollouts": 38, "validation_rollouts": 3})
                    models.append((name, feature, seed, model))
                    (out / "training.json").write_text(json.dumps(training, indent=2) + "\n")
        # Test arrays are first loaded only after all fits in the family are complete.
        for split in SPLITS[2:]:
            data = make_windows(load_split(family, split), args.stride)
            predictions = {"persistence": np.repeat(data["current"][:, None, :], len(HORIZONS), axis=1), "constant_velocity": data["cv"]}
            for name, pred in predictions.items():
                records.extend(score(pred, data, family=family, split=split, name=name, seed=0))
            for name, feature, seed, model in models:
                pred = model.predict(data["features"][feature]).reshape(-1, len(HORIZONS), 3) + data["cv"]
                predictions[f"{name}_seed{seed}"] = pred
                records.extend(score(pred, data, family=family, split=split, name=name, seed=seed))
            np.savez_compressed(out / f"{family}_{split}_predictions.npz", target=data["target"], rollout=data["rollout"], anchor=data["anchor"], **predictions)
            (out / "metrics_by_rollout.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"COMPLETE {family}: {time.perf_counter()-start:.1f}s", flush=True)
    summary=[]
    for family in args.families:
        for split in SPLITS[2:]:
            for name in config["models"]:
                for h in HORIZONS:
                    rows=[r for r in records if r["family"]==family and r["split"]==split and r["model"]==name and r["horizon_ms"]==h]
                    summary.append({"family":family,"split":split,"model":name,"horizon_ms":int(h),"rmse_rad":float(np.sqrt(np.mean([r["mse_rad2"] for r in rows]))),"n_rollouts":9,"n_training_seeds":len(set(r["seed"] for r in rows))})
    (out / "summary.json").write_text(json.dumps(summary,indent=2)+'\n')
    print(f"Saved {len(records)} rollout metrics to {out}",flush=True)


if __name__ == "__main__":
    main()
