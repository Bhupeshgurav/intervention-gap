"""NumPy adapter of the independently authored PC-Gym CSTR equations.

Physical equations and defaults are adapted from PC-Gym's ``cstr`` at
fbfb36bf84b494cd157cff88292e1fd1b864cee5 (MIT, Copyright 2024 Maximilian Bloor).
See vendor/pcgym_reference/LICENSE and source_manifest.json. We do not import
PC-Gym or reproduce its environment, normalization, reward, or solver.

State: [Ca, T]. Input: jacket temperature Tc. Disturbances: Ti and Caf.
Temperatures are kelvin. Time/concentration use the upstream numerical units;
the linked original reference has inconsistent labels. See README.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from math import ceil
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CSTRParameters:
    q: float = 100.0
    V: float = 100.0
    rho: float = 1000.0
    C: float = 0.239
    deltaHr: float = -5e4
    EA_over_R: float = 8750.0
    k0: float = 7.2e10
    UA: float = 5e4
    Ti: float = 350.0
    Caf: float = 1.0

    def __post_init__(self):
        values = np.array([getattr(self, f.name) for f in fields(self)])
        if not np.all(np.isfinite(values)):
            raise ValueError("CSTR parameters must be finite")
        if min(self.q, self.V, self.rho, self.C, self.EA_over_R,
               self.k0, self.UA, self.Ti) <= 0 or self.Caf < 0:
            raise ValueError("Invalid CSTR physical parameters")


DEFAULT_PARAMETERS = CSTRParameters()
REFERENCE_EQUILIBRIUM = np.array([0.87725294608097, 324.475443431599])


def cstr_rhs(state, coolant, *, params=DEFAULT_PARAMETERS,
             inlet_temperature=None, inlet_concentration=None):
    """Vectorized exact upstream derivative; ``state.shape == (..., 2)``.

    Coolant and disturbances broadcast against ``state.shape[:-1]``.
    No clipping, normalization, or extra plant response rule is applied.
    """
    x = np.asarray(state, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != 2:
        raise ValueError("state must have shape (..., 2)")
    ca, temperature = x[..., 0], x[..., 1]
    tc = np.asarray(coolant, dtype=np.float64)
    ti = params.Ti if inlet_temperature is None else np.asarray(inlet_temperature)
    caf = params.Caf if inlet_concentration is None else np.asarray(inlet_concentration)
    if np.any(temperature <= 0):
        raise FloatingPointError("The CSTR temperature must remain positive")
    reaction_rate = params.k0 * np.exp(-params.EA_over_R / temperature) * ca
    dcadt = params.q / params.V * (caf - ca) - reaction_rate
    dtdt = (
        params.q / params.V * (ti - temperature)
        + (-params.deltaHr) * reaction_rate / (params.rho * params.C)
        + params.UA * (tc - temperature) / (params.rho * params.C * params.V)
    )
    dcadt, dtdt = np.broadcast_arrays(dcadt, dtdt)
    return np.stack((dcadt, dtdt), axis=-1)


def integrate_interval(state, coolant, *, dt=0.1, max_step=0.01,
                       params=DEFAULT_PARAMETERS, inlet_temperature=None,
                       inlet_concentration=None):
    """Fixed-step RK4 under a zero-order-held input and disturbances.

    State is never clipped to make a trajectory appear valid. Nonfinite or
    unphysical numerical states raise an error rather than being repaired.
    Numerical convergence must be checked again outside the smoke-test domain.
    """
    if not np.isfinite(dt) or not np.isfinite(max_step) or dt <= 0 or max_step <= 0:
        raise ValueError("dt and max_step must be finite and positive")
    x = np.array(state, dtype=np.float64, copy=True)
    if x.ndim < 1 or x.shape[-1] != 2 or not np.all(np.isfinite(x)):
        raise ValueError("state must be finite with shape (..., 2)")
    inputs = [coolant]
    inputs += [v for v in (inlet_temperature, inlet_concentration) if v is not None]
    if any(not np.all(np.isfinite(v)) for v in inputs):
        raise ValueError("Inputs and disturbances must be finite")
    n = max(1, int(ceil(dt / max_step)))
    h = dt / n
    kw = dict(params=params, inlet_temperature=inlet_temperature,
              inlet_concentration=inlet_concentration)
    for _ in range(n):
        k1 = cstr_rhs(x, coolant, **kw)
        k2 = cstr_rhs(x + h * k1 / 2, coolant, **kw)
        k3 = cstr_rhs(x + h * k2 / 2, coolant, **kw)
        k4 = cstr_rhs(x + h * k3, coolant, **kw)
        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not np.all(np.isfinite(x)) or np.any(x[..., 0] < 0) or np.any(x[..., 1] <= 0):
            raise FloatingPointError("Invalid RK4 state; reduce step or inspect operating domain")
    return x


def simulate_commands(initial_state, commands, *, dt=0.1, max_step=0.01,
                      params=DEFAULT_PARAMETERS, inlet_temperature=None,
                      inlet_concentration=None):
    """Batch zero-order-held *applied* command rollouts.

    ``initial_state`` is [batch, 2] or [2]. ``commands`` is [batch, steps] or
    [steps] (broadcast to every batch member). Disturbances accept scalars,
    [steps], or [batch, steps]. Output is [batch, steps + 1, 2], including x0;
    a scalar initial state instead returns [steps + 1, 2].

    This low-level API intentionally has no command delay/clipping wrapper.
    To model the request-to-execution channel, use CSTRSimulator below or
    construct the applied-input sequence explicitly and log both sequences.
    """
    x0 = np.asarray(initial_state, dtype=np.float64)
    single = x0.ndim == 1
    if single:
        x0 = x0[None, :]
    if x0.ndim != 2 or x0.shape[1] != 2:
        raise ValueError("initial_state must have shape [batch, 2] or [2]")
    u = np.asarray(commands, dtype=np.float64)
    if u.ndim not in (1, 2):
        raise ValueError("commands must have shape [steps] or [batch, steps]")
    u = np.broadcast_to(u, (len(x0), u.shape[-1]))
    shape = u.shape
    ti = np.broadcast_to(params.Ti if inlet_temperature is None else inlet_temperature, shape)
    caf = np.broadcast_to(params.Caf if inlet_concentration is None else inlet_concentration, shape)
    out = np.empty((len(x0), shape[1] + 1, 2), dtype=np.float64)
    out[:, 0] = x0
    for k in range(shape[1]):
        out[:, k + 1] = integrate_interval(
            out[:, k], u[:, k], dt=dt, max_step=max_step, params=params,
            inlet_temperature=ti[:, k], inlet_concentration=caf[:, k],
        )
    return out[0] if single else out


@dataclass(frozen=True)
class ObservationSpec:
    """Explicit learner-visible fields; hidden variables are omitted, not zeroed."""

    visible: tuple[str, ...] = ("Ca", "T")

    def __post_init__(self):
        allowed = {"Ca", "T", "Ti", "Caf", "applied_Tc"}
        if len(set(self.visible)) != len(self.visible) or not set(self.visible) <= allowed:
            raise ValueError(f"Unique observation fields required, drawn from {sorted(allowed)}")


@dataclass(frozen=True)
class ExogenousTape:
    """Pre-sampled disturbances and measurement noise, indexed by physical step.

    Feed arrays have steps + 1 entries, noise has [steps + 1, 2]. They are
    copied and made read-only. Branches share the same tape. Randomness is
    never drawn inside step(), so different action choices cannot consume
    or shift the disturbance stream. No arbitrary process noise is added.
    """

    inlet_temperature: np.ndarray
    inlet_concentration: np.ndarray
    measurement_noise: np.ndarray

    def __post_init__(self):
        ti = np.array(self.inlet_temperature, dtype=np.float64, copy=True)
        caf = np.array(self.inlet_concentration, dtype=np.float64, copy=True)
        noise = np.array(self.measurement_noise, dtype=np.float64, copy=True)
        if ti.ndim != 1 or len(ti) < 2 or caf.shape != ti.shape or noise.shape != (len(ti), 2):
            raise ValueError("Tape shapes must be [steps+1], [steps+1], [steps+1,2]")
        if any(not np.all(np.isfinite(a)) for a in (ti, caf, noise)):
            raise ValueError("Exogenous tape must be finite")
        if np.any(ti <= 0) or np.any(caf < 0):
            raise ValueError("Inlet temperatures must be positive and concentrations nonnegative")
        for key, a in (("inlet_temperature", ti), ("inlet_concentration", caf),
                       ("measurement_noise", noise)):
            a.setflags(write=False)
            object.__setattr__(self, key, a)

    @property
    def steps(self):
        return len(self.inlet_temperature) - 1

    @classmethod
    def constant(cls, steps, *, inlet_temperature=350.0, inlet_concentration=1.0):
        return cls(np.full(steps + 1, inlet_temperature),
                   np.full(steps + 1, inlet_concentration), np.zeros((steps + 1, 2)))

    @classmethod
    def sampled(cls, steps, *, seed, inlet_temperature=350.0, inlet_concentration=1.0,
                temperature_std=0.0, concentration_std=0.0,
                measurement_std=(0.0, 0.0)):
        """Independent white feed perturbations: a documented wrapper assumption.

        Choose small perturbations; invalid feeds raise, rather than clipping.
        Initial-state and policy RNGs should use separate experiment seeds.
        """
        if min(temperature_std, concentration_std, *measurement_std) < 0:
            raise ValueError("Standard deviations must be nonnegative")
        seeds = np.random.SeedSequence(seed).spawn(3)
        ti = inlet_temperature + np.random.default_rng(seeds[0]).normal(0, temperature_std, steps + 1)
        caf = inlet_concentration + np.random.default_rng(seeds[1]).normal(0, concentration_std, steps + 1)
        noise = np.random.default_rng(seeds[2]).normal(size=(steps + 1, 2)) * measurement_std
        return cls(ti, caf, noise)


@dataclass(frozen=True)
class ExecutionConfig:
    lower: float = 290.0
    upper: float = 302.0
    delay_steps: int = 0
    initial_applied: float = 300.0

    def __post_init__(self):
        if not np.all(np.isfinite([self.lower, self.upper, self.initial_applied])):
            raise ValueError("Execution bounds must be finite")
        if self.lower >= self.upper or not self.lower <= self.initial_applied <= self.upper:
            raise ValueError("Invalid execution limits or initial applied input")
        if not isinstance(self.delay_steps, int) or self.delay_steps < 0:
            raise ValueError("delay_steps must be a nonnegative integer")


class CSTRSimulator:
    """Stateful adapter with clone/reset and separate public/privileged views.

    Bounds are command limits, not a proof of process safety. Returned flags
    record simulated operating-envelope violations without clipping the state.
    """

    def __init__(self, *, params=DEFAULT_PARAMETERS, dt=0.1, max_step=0.01,
                 observation=ObservationSpec(), execution=ExecutionConfig(),
                 temperature_limits=(300.0, 350.0)):
        self.params = params
        self.dt = float(dt)
        self.max_step = float(max_step)
        self.observation = observation
        self.execution = execution
        self.temperature_limits = tuple(temperature_limits)
        if self.dt <= 0 or self.max_step <= 0 or not np.all(np.isfinite([self.dt, self.max_step])):
            raise ValueError("Positive finite integration intervals required")
        if len(self.temperature_limits) != 2 or self.temperature_limits[0] >= self.temperature_limits[1]:
            raise ValueError("temperature_limits must be an ordered pair")
        self._state = None
        self._tape = None

    def reset(self, initial_state=(0.8, 330.0), *, tape=None):
        x = np.asarray(initial_state, dtype=np.float64)
        if x.shape != (2,) or not np.all(np.isfinite(x)) or x[0] < 0 or x[1] <= 0:
            raise ValueError("initial_state must be a finite physical [Ca,T] pair")
        self._state = x.copy()
        self._initial_state = x.copy()
        self._tape = tape if tape is not None else ExogenousTape.constant(
            100, inlet_temperature=self.params.Ti, inlet_concentration=self.params.Caf)
        self._index = 0
        self._applied = self.execution.initial_applied
        self._queue = deque([(self._applied, None)] * self.execution.delay_steps)
        return self.observe()

    def _ready(self):
        if self._state is None:
            raise RuntimeError("Call reset() before operating the simulator")

    def observe(self):
        self._ready()
        noisy = self._state + self._tape.measurement_noise[self._index]
        values = {"Ca": noisy[0], "T": noisy[1],
                  "Ti": self._tape.inlet_temperature[self._index],
                  "Caf": self._tape.inlet_concentration[self._index],
                  "applied_Tc": self._applied}
        return {name: float(values[name]) for name in self.observation.visible}

    def evaluator_state(self):
        """Privileged state for evaluation ONLY; never merge into learner logs."""
        self._ready()
        return {"state": self._state.copy(), "step": self._index,
                "inlet_temperature": float(self._tape.inlet_temperature[self._index]),
                "inlet_concentration": float(self._tape.inlet_concentration[self._index]),
                "pending_execution": tuple(self._queue)}

    def clone(self):
        """Branch at exactly the same physical state, execution queue and tape index."""
        self._ready()
        other = CSTRSimulator(params=self.params, dt=self.dt, max_step=self.max_step,
                              observation=self.observation, execution=self.execution,
                              temperature_limits=self.temperature_limits)
        other._state = self._state.copy()
        other._initial_state = self._initial_state.copy()
        other._tape = self._tape
        other._index = self._index
        other._applied = self._applied
        other._queue = deque(self._queue)
        return other

    def step(self, requested_coolant):
        self._ready()
        if self._index >= self._tape.steps:
            raise IndexError("Exogenous tape exhausted")
        requested = float(requested_coolant)
        if not np.isfinite(requested):
            raise ValueError("Requested coolant must be finite")
        clipped = float(np.clip(requested, self.execution.lower, self.execution.upper))
        queue = deque(self._queue)
        queue.append((clipped, self._index))
        applied, source_step = queue.popleft()
        before = self.observe()
        next_state = integrate_interval(
            self._state, applied, dt=self.dt, max_step=self.max_step, params=self.params,
            inlet_temperature=self._tape.inlet_temperature[self._index],
            inlet_concentration=self._tape.inlet_concentration[self._index],
        )
        k = self._index
        self._state, self._queue, self._applied = next_state, queue, applied
        self._index += 1
        lo, hi = self.temperature_limits
        return {
            "step": k, "time": k * self.dt,
            "observation": before, "next_observation": self.observe(),
            "requested_Tc": requested, "clipped_request_Tc": clipped,
            "request_clipped": clipped != requested,
            "applied_Tc": applied, "applied_request_step": source_step,
            "applied_at_time": k * self.dt,
            "temperature_limit_violated": not lo <= next_state[1] <= hi,
        }


@dataclass
class PIController:
    """Conventional temperature PI control with conditional anti-windup.

    An operational baseline, not a globally optimal controller. It needs a
    measured reactor temperature and its gains must be fixed before testing.
    """

    target: float = 324.475443431599
    bias: float = 300.0
    kp: float = 1.0
    ki: float = 0.5
    lower: float = 290.0
    upper: float = 302.0
    integral: float = 0.0

    def reset(self):
        self.integral = 0.0

    def __call__(self, measured_temperature, *, dt=0.1):
        error = self.target - float(measured_temperature)
        candidate = self.integral + error * dt
        raw = self.bias + self.kp * error + self.ki * candidate
        if (self.lower <= raw <= self.upper or
                (raw > self.upper and error < 0) or
                (raw < self.lower and error > 0)):
            self.integral = candidate
        return float(np.clip(self.bias + self.kp * error + self.ki * self.integral,
                             self.lower, self.upper))
