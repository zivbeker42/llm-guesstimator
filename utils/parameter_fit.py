"""Parameter fitting helpers for prefill/decode estimators."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .config import HardwareConfig, ModelConfig
from .math_utils import (
    decode_compute_time,
    decode_memory_time,
    total_prefill_time,
    prefill_memory_time,
)

TUNABLE_PARAMETER_NAMES: Dict[str, Iterable[str]] = {
    "total_prefill_time": ["running_tokens_cap", "prefill_mult_factor"],
    "decode_memory_time": ["running_tokens_cap", "decode_time_min"],
}


def fit_prefill_decode_parameters(
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    hardware_cfg: HardwareConfig,
    scenario_cfg: Any,
    tunable_params: Dict[str, Iterable[str]],
    parameter_bounds: Dict[Tuple[str, str] | str, Tuple[float, float]] | None = None,
    max_iter: int = 200,
):
    """Fit tuning parameters to TTFT/ITL observations via least squares."""

    parameter_bounds = parameter_bounds or {}
    default_bounds = {
        "running_tokens_cap": (float(max(df["input_tokens"].max(), 1e4)), 1e8),
        "prefill_mult_factor": (0.1, 10.0),
        "decode_time_min": (0.0, 0.01)
    }

    prefill_defaults = {
        "running_tokens_cap": float(getattr(scenario_cfg, "prefill_running_tokens_cap")),
        "prefill_mult_factor": 1.0,
    }
    decode_defaults = {
        "running_tokens_cap": float(getattr(scenario_cfg, "decode_running_tokens_cap")),
        "decode_time_min": 0.0
    }

    S = df["concurrency"].to_numpy(dtype=float)
    L_prompt = df["input_tokens"].to_numpy(dtype=float)
    L_decode = L_prompt + df["output_tokens"].to_numpy(dtype=float)
    ttft_actual = df["TTFT_ms"].to_numpy(dtype=float)
    itl_actual = df["ITL_ms"].to_numpy(dtype=float)

    valid_functions = set(TUNABLE_PARAMETER_NAMES)
    for func_name in tunable_params:
        if func_name not in valid_functions:
            raise ValueError(f"Unsupported function '{func_name}'.")

    param_specs = []
    theta0 = []
    theta_lower = []
    theta_upper = []

    def resolve_bounds(func_name: str, param_name: str) -> Tuple[float, float]:
        key = (func_name, param_name)
        if key in parameter_bounds:
            return parameter_bounds[key]
        if param_name in parameter_bounds:
            return parameter_bounds[param_name]
        return default_bounds[param_name]

    for func_name, param_list in tunable_params.items():
        defaults = prefill_defaults if func_name == "total_prefill_time" else decode_defaults
        for param_name in param_list:
            if param_name not in defaults:
                raise ValueError(f"Cannot tune '{param_name}' for {func_name}.")
            baseline_value = float(defaults[param_name])
            lower, upper = resolve_bounds(func_name, param_name)
            lower = float(lower)
            upper = float(upper)
            if not lower < upper:
                raise ValueError(f"Invalid bounds for {func_name}.{param_name}.")
            init_value = float(np.clip(baseline_value, lower * 1.01, upper * 0.99))
            param_specs.append(
                {
                    "function": func_name,
                    "name": param_name,
                    "baseline": baseline_value,
                    "lower": lower,
                    "upper": upper,
                }
            )
            theta0.append(np.log(init_value))
            theta_lower.append(np.log(lower))
            theta_upper.append(np.log(upper))

    def unpack_theta(theta):
        overrides = {"total_prefill_time": {}, "decode_memory_time": {}}
        for spec, value in zip(param_specs, theta):
            overrides[spec["function"]][spec["name"]] = float(np.exp(value))
        return overrides

    def build_params(overrides: Dict[str, Dict[str, float]]):
        prefill_params = prefill_defaults.copy()
        prefill_params.update(overrides.get("total_prefill_time", {}))
        decode_params = decode_defaults.copy()
        decode_params.update(overrides.get("decode_memory_time", {}))
        return prefill_params, decode_params

    def compute_predictions(overrides: Dict[str, Dict[str, float]]):
        prefill_params, decode_params = build_params(overrides)
        prefill_compute = total_prefill_time(
            S,
            L_prompt,
            model_cfg,
            hardware_cfg,
            **prefill_params,
        )
        prefill_memory = prefill_memory_time(S, L_prompt, model_cfg, hardware_cfg)
        decode_compute = decode_compute_time(S, L_decode, model_cfg, hardware_cfg)
        decode_memory = decode_memory_time(
            S,
            L_decode,
            model_cfg,
            hardware_cfg,
            **decode_params,
        )
        ttft_pred_ms = np.maximum(prefill_compute, prefill_memory) * 1e3
        itl_pred_ms = np.maximum(decode_compute, decode_memory) * 1e3
        return ttft_pred_ms, itl_pred_ms

    def compute_loss(overrides: Dict[str, Dict[str, float]]):
        ttft_pred_ms, itl_pred_ms = compute_predictions(overrides)
        ttft_rel = (ttft_pred_ms - ttft_actual) / np.maximum(ttft_actual, 1e-6)
        itl_rel = (itl_pred_ms - itl_actual) / np.maximum(itl_actual, 1e-6)
        residuals = np.concatenate([ttft_rel, itl_rel])
        return float(np.mean(np.square(residuals)))

    if not param_specs:
        baseline_overrides = {"total_prefill_time": {}, "decode_memory_time": {}}
        baseline_params = build_params(baseline_overrides)
        parameter_summary = pd.DataFrame(columns=["function", "parameter", "baseline", "optimized", "delta_pct"])
        error_summary = pd.DataFrame(
            [
                {
                    "label": "baseline",
                    "loss": compute_loss(baseline_overrides),
                }
            ]
        )
        history = pd.DataFrame([[0, error_summary.loc[0, "loss"]]], columns=["iteration", "loss"])
        best_params = {
            "total_prefill_time": baseline_params[0],
            "decode_memory_time": baseline_params[1],
        }
        return best_params, {
            "parameter_summary": parameter_summary,
            "error_summary": error_summary,
            "loss_history": history,
        }

    theta0 = np.array(theta0)
    theta_lower = np.array(theta_lower)
    theta_upper = np.array(theta_upper)

    def loss_from_theta(theta):
        bounded = np.clip(theta, theta_lower, theta_upper)
        return compute_loss(unpack_theta(bounded))

    baseline_loss = loss_from_theta(theta0)
    history = [(0, baseline_loss)]

    def numerical_gradient(theta, eps=1e-3):
        grad = np.zeros_like(theta)
        for idx in range(theta.size):
            delta = np.zeros_like(theta)
            delta[idx] = eps
            grad[idx] = (loss_from_theta(theta + delta) - loss_from_theta(theta - delta)) / (2 * eps)
        return grad

    theta = theta0.copy()
    current_loss = baseline_loss
    step = 0.1

    for iteration in range(1, max_iter + 1):
        grad = numerical_gradient(theta)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-6:
            break

        step_candidate = step
        improved = False
        for _ in range(8):
            candidate = np.clip(theta - step_candidate * grad, theta_lower, theta_upper)
            candidate_loss = loss_from_theta(candidate)
            if candidate_loss < current_loss:
                theta = candidate
                current_loss = candidate_loss
                history.append((iteration, current_loss))
                step = step_candidate * 1.1
                improved = True
                break
            step_candidate *= 0.5

        if not improved:
            step *= 0.5
            if step < 1e-5:
                break

    best_overrides = unpack_theta(theta)
    best_prefill, best_decode = build_params(best_overrides)
    best_params = {
        "total_prefill_time": best_prefill,
        "decode_memory_time": best_decode,
    }

    parameter_rows = []
    for spec in param_specs:
        optimized_value = best_params[spec["function"]][spec["name"]]
        baseline_value = spec["baseline"]
        delta_pct = ((optimized_value - baseline_value) / baseline_value) * 100 if baseline_value else np.nan
        parameter_rows.append(
            {
                "function": spec["function"],
                "parameter": spec["name"],
                "baseline": baseline_value,
                "optimized": optimized_value,
                "delta_pct": delta_pct,
            }
        )
    parameter_summary = pd.DataFrame(parameter_rows)

    def build_error_summary(overrides, label):
        ttft_pred_ms, itl_pred_ms = compute_predictions(overrides)
        ttft_rel = (ttft_pred_ms - ttft_actual) / np.maximum(ttft_actual, 1e-6)
        itl_rel = (itl_pred_ms - itl_actual) / np.maximum(itl_actual, 1e-6)
        return {
            "label": label,
            "loss": compute_loss(overrides),
            "ttft_mape_pct": float(np.mean(np.abs(ttft_rel)) * 100),
            "ttft_rmse_ms": float(np.sqrt(np.mean((ttft_pred_ms - ttft_actual) ** 2))),
            "itl_mape_pct": float(np.mean(np.abs(itl_rel)) * 100),
            "itl_rmse_ms": float(np.sqrt(np.mean((itl_pred_ms - itl_actual) ** 2))),
        }

    error_summary = pd.DataFrame(
        [
            build_error_summary({"total_prefill_time": {}, "decode_memory_time": {}}, "baseline"),
            build_error_summary(best_overrides, "optimized"),
        ]
    )

    history_df = pd.DataFrame(history, columns=["iteration", "loss"])

    return best_params, {
        "parameter_summary": parameter_summary,
        "error_summary": error_summary,
        "loss_history": history_df,
    }
