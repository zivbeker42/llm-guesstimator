"""Shared analytical helpers for the estimator notebooks."""

from __future__ import annotations

import numpy as np

from .config import HardwareConfig, ModelConfig


def per_request_per_layer_flops_kv(L, model: ModelConfig) -> np.ndarray:
    """Per-request, per-layer FLOPs for decoding with a KV cache."""

    d = model.hidden_size
    r = model.expansion_ratio
    return (8 + 4 * r) * d**2 + 4 * L * d


def server_total_per_layer_flops(
    L, total_prompt_tokens: float, model: ModelConfig
) -> np.ndarray:
    """Total per-layer FLOPs across requests with P = S * L fixed."""

    d = model.hidden_size
    r = model.expansion_ratio
    P = total_prompt_tokens
    return P * ((8 + 4 * r) * d**2) / L + 4 * P * d


def server_total_model_flops(
    L, total_prompt_tokens: float, model: ModelConfig
) -> np.ndarray:
    """Total model FLOPs across requests (per decode token)."""

    return (
        server_total_per_layer_flops(L, total_prompt_tokens, model) * model.num_layers
    )


def time_per_token_from_L(
    L,
    total_prompt_tokens: float,
    model: ModelConfig,
    hardware: HardwareConfig,
) -> np.ndarray:
    """Decode time per new token as a function of context length."""

    flops = server_total_model_flops(L, total_prompt_tokens, model)
    return flops / (hardware.flops_per_second * hardware.gpu_count)


def time_per_token_from_S(
    S,
    total_prompt_tokens: float,
    model: ModelConfig,
    hardware: HardwareConfig,
) -> np.ndarray:
    """Decode time per new token as a function of concurrency."""

    L = total_prompt_tokens / S
    return time_per_token_from_L(L, total_prompt_tokens, model, hardware)


def decode_compute_time(
    S,
    L,
    model: ModelConfig,
    hardware: HardwareConfig,
) -> np.ndarray:
    """Decode compute time per generated token."""

    d = model.hidden_size
    r = model.expansion_ratio
    n_layers = model.num_layers
    F = hardware.flops_per_second * hardware.gpu_count
    return (n_layers * S * ((8 + 4 * r) * d**2 + 4 * L * d)) / F


def decode_memory_time(
    S,
    L,
    model: ModelConfig,
    hardware: HardwareConfig,
    running_tokens_cap: int = np.inf,
    decode_time_min: float = 0,
    bytes_mult_factor: float = 1,
) -> np.ndarray:
    """Decode memory time per generated token."""

    d = model.hidden_size
    r = model.expansion_ratio
    n_layers = model.num_layers
    dtype_bytes = hardware.dtype_bytes
    BW = hardware.memory_bandwidth * hardware.gpu_count
    c_act = hardware.activation_io_multiplier
    S_eff = np.minimum(S, running_tokens_cap / L)
    bytes_total = (
        n_layers
        * (((4 + 2 * r) * d**2) + (2 * S_eff * L * d) + ((2 + c_act) * S_eff * d))
        * dtype_bytes
    ) * bytes_mult_factor  # a factor to make up wrong math
    return bytes_total / BW + decode_time_min


def total_prefill_time(
    S,
    L,
    model: ModelConfig,
    hardware: HardwareConfig,
    running_tokens_cap: int = None,
    prefill_mult_factor: float = 1,
    offload_mult_factor: float = 1,
) -> np.ndarray:
    return (
        prefill_compute_time(
            S,
            L,
            model,
            hardware,
            running_tokens_cap,
            prefill_mult_factor,
        )
        + prefill_memory_HBM_wall_time(S, L, model, hardware) * offload_mult_factor
        + prefill_tensor_parallel_time(S, L, model, hardware)
    )


def prefill_compute_time(
    S,
    L,
    model: ModelConfig,
    hardware: HardwareConfig,
    running_tokens_cap: int = None,
    prefill_mult_factor: float = 1,
) -> np.ndarray:
    """Prefill compute time per batch step."""

    d = model.hidden_size
    r = model.expansion_ratio
    n_layers = model.num_layers
    running_tokens_cap = running_tokens_cap if running_tokens_cap else S * L
    num_micro_batches = np.ceil(S * L / running_tokens_cap)
    F = hardware.flops_per_second * hardware.gpu_count

    pure_compute_time = (
        n_layers
        * S  # (1 + ((num_micro_batches - 1) * num_micro_batches)) # goes quadratic as every micro-batch waits for his own turn
        * ((8 + 4 * r) * L * d**2 + 4 * (L**2) * d)
        / F
    )
    return (
        prefill_mult_factor * pure_compute_time
    )  # prefill_mult_factor mearnt to fit best to benchmark results


def prefill_memory_HBM_wall_time(
    S: int, L: float, model: ModelConfig, hardware: HardwareConfig
):
    """extra time it takes if memory is filled and need swapping with host main memory"""
    d = model.hidden_size
    r = model.expansion_ratio
    n_layers = model.num_layers
    dtype_bytes = hardware.dtype_bytes
    model_weights_size = model.model_size * dtype_bytes
    kv_cache_size = n_layers * 2 * S * L * d * dtype_bytes
    # print(f"{S=}, {L=}, {kv_cache_size/1e9=}GB")
    PCIe_BW_one_way = (
        hardware.PCIe_bandwidth * hardware.gpu_count / 2
    )  # dividing by 2 as PCIe specs refers to 2 way communication

    bytes_swapped = np.maximum(
        0, model_weights_size + kv_cache_size - hardware.HBM_size * hardware.gpu_count
    )
    return bytes_swapped / PCIe_BW_one_way


def prefill_memory_time(
    S,
    L,
    model: ModelConfig,
    hardware: HardwareConfig,
) -> np.ndarray:
    """Prefill memory time per batch step."""

    d = model.hidden_size
    r = model.expansion_ratio
    n_layers = model.num_layers
    dtype_bytes = hardware.dtype_bytes
    BW = hardware.memory_bandwidth * hardware.gpu_count
    c_act = hardware.activation_io_multiplier
    bytes_total = (
        n_layers * ((4 + 2 * r) * d**2 + (2 + c_act) * S * L * d) * dtype_bytes
    )
    return bytes_total / BW


def prefill_tensor_parallel_time(
    S: float,
    L: float,
    model: ModelConfig,
    hardware: HardwareConfig,
):
    d = model.hidden_size
    n_layers = model.num_layers
    dtype_bytes = hardware.dtype_bytes
    parallelism = hardware.gpu_count
    bandwidth = (
        hardware.NVLINK_bandwidth / 2
    )  # specs are for 2-way direction, and wer'e interested in one-direction BW.
    return (
        (3 * (parallelism - 1) / parallelism)
        * n_layers
        * S
        * L
        * d
        * dtype_bytes
        / bandwidth
    )


def weights_bytes(model: ModelConfig, hardware: HardwareConfig) -> float:
    """Bytes needed to store the model weights."""

    d = model.hidden_size
    r = model.expansion_ratio
    n_layers = model.num_layers
    return n_layers * (4 + 2 * r) * d**2 * hardware.dtype_bytes


def kv_read_bytes(
    total_prompt_tokens: float, model: ModelConfig, hardware: HardwareConfig
) -> float:
    """Bytes read from the KV cache during decode."""

    d = model.hidden_size
    n_layers = model.num_layers
    return n_layers * (2 * total_prompt_tokens * d) * hardware.dtype_bytes


def kv_write_bytes(
    concurrency: float, model: ModelConfig, hardware: HardwareConfig
) -> float:
    """Bytes written to the KV cache during decode."""

    d = model.hidden_size
    n_layers = model.num_layers
    return n_layers * (2 * concurrency * d) * hardware.dtype_bytes


def activation_io_bytes(
    concurrency: float, model: ModelConfig, hardware: HardwareConfig
) -> float:
    """Activation I/O bytes exchanged during decode."""

    d = model.hidden_size
    n_layers = model.num_layers
    c_act = hardware.activation_io_multiplier
    return n_layers * (c_act * concurrency * d) * hardware.dtype_bytes


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return numerator / denominator, guarding against divide-by-zero."""

    denominator = np.where(denominator == 0, 1, denominator)
    return numerator / denominator
