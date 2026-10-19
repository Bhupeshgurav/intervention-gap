"""Small supervised predictors with train-only scaling and CPU checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass
class TorchRegressor:
    seed: int = 0
    epochs: int = 160
    width: int = 64
    batch_size: int = 256
    learning_rate: float = 0.002
    weight_decay: float = 0.0001

    def fit(self, x, y, *, validation=None):
        start = time.perf_counter()
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        self.x_mean, self.x_scale = x.mean(0), x.std(0).clip(1e-6)
        self.y_mean, self.y_scale = y.mean(0), y.std(0).clip(1e-6)
        xt = torch.from_numpy((x - self.x_mean) / self.x_scale)
        yt = torch.from_numpy((y - self.y_mean) / self.y_scale)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], self.width), torch.nn.Tanh(),
            torch.nn.Linear(self.width, self.width), torch.nn.Tanh(),
            torch.nn.Linear(self.width, y.shape[1]),
        )
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.history = []
        self.best_epoch = self.epochs
        best_loss, best_weights = float("inf"), None
        if validation is not None:
            vx, vy = validation
            vx = torch.from_numpy(np.asarray((vx - self.x_mean) / self.x_scale, dtype=np.float32))
            vy = torch.from_numpy(np.asarray((vy - self.y_mean) / self.y_scale, dtype=np.float32))
        for epoch in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(x))
            loss_sum = 0.0
            for begin in range(0, len(x), self.batch_size):
                batch = order[begin:begin + self.batch_size]
                loss = torch.mean((self.net(xt[batch]) - yt[batch]) ** 2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach()) * len(batch)
            record = {"epoch": epoch + 1, "training_standardized_mse": loss_sum / len(x)}
            if validation is not None:
                self.net.eval()
                with torch.no_grad():
                    score = float(torch.mean((self.net(vx) - vy) ** 2))
                record["validation_standardized_mse"] = score
                if score < best_loss:
                    best_loss = score
                    self.best_epoch = epoch + 1
                    best_weights = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            self.history.append(record)
        if best_weights is not None:
            self.net.load_state_dict(best_weights)
        self.fit_seconds = time.perf_counter() - start
        self.parameter_count = sum(p.numel() for p in self.net.parameters())
        return self

    def predict(self, x):
        self.net.eval()
        xt = torch.from_numpy(np.asarray((x - self.x_mean) / self.x_scale, dtype=np.float32))
        with torch.no_grad():
            pred = torch.cat([self.net(part) for part in xt.split(8192)]).numpy()
        return pred * self.y_scale + self.y_mean

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "x_mean": torch.from_numpy(self.x_mean), "x_scale": torch.from_numpy(self.x_scale),
            "y_mean": torch.from_numpy(self.y_mean), "y_scale": torch.from_numpy(self.y_scale),
            "seed": self.seed, "epochs": self.epochs, "best_epoch": self.best_epoch,
            "width": self.width, "history": self.history,
            "parameter_count": self.parameter_count,
        }, path)


def fit_predictor(kind, x, y, *, seed=0, epochs=160, validation=None):
    if kind == "mlp":
        return TorchRegressor(seed=seed, epochs=epochs).fit(x, y, validation=validation)
    if kind == "ridge":
        model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=1.0))
    elif kind == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=3, random_state=seed, n_jobs=2)
    else:
        raise ValueError(f"Unknown predictor {kind}")
    start = time.perf_counter()
    model.fit(x, y)
    model.fit_seconds = time.perf_counter() - start
    return model
