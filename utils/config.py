"""Central configuration for hardware, model, and workload defaults."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class HardwareConfig:
    """Describes accelerator characteristics relevant to the estimators."""

    flops_per_second: float
    memory_bandwidth: float
    dtype_bytes: float
    activation_io_multiplier: float
    PCIe_bandwidth: int
    HBM_size: int 
    gpu_count: int = 1


@dataclass(frozen=True)
class ModelConfig:
    """Represents transformer model dimensions used in the estimators."""

    hidden_size: int
    num_layers: int
    expansion_ratio: float
    model_size: float


@dataclass(frozen=True)
class WorkloadConfig:
    """Scenario-specific knobs for the analytical calculations."""

    total_prompt_tokens: float


DEFAULT_HARDWARE_NAME = "A100_40GB_FP16"
DEFAULT_MODEL_NAME = "70B"
DEFAULT_DECODE_MODEL_NAME = "decode_example"

HARDWARE_PRESETS: Dict[str, HardwareConfig] = {
    DEFAULT_HARDWARE_NAME: HardwareConfig(
        flops_per_second=3.12e14,  # sustained FLOPs/s (A100 FP16/BF16)
        memory_bandwidth=1.555e12,  # sustained memory bandwidth (bytes/s)
        dtype_bytes=2.0,
        activation_io_multiplier=12.0,
        PCIe_bandwidth=120e9,
        HBM_size=8.59e10
    ),
    "H100_80GB_FP8_TP2": HardwareConfig(
        flops_per_second=3.958e15,  # NVIDIA H100 SXM FP8 tensor throughput per GPU
        memory_bandwidth=3.35e12,  # sustained HBM3 bandwidth per GPU (bytes/s)
        dtype_bytes=1.0,
        activation_io_multiplier=12.0,
        PCIe_bandwidth=120e9,
        HBM_size=8.59e10,
        gpu_count=2,
        
    ),
    "H100_80GB_FP16_TP4": HardwareConfig(
        flops_per_second=1.979e15,  # NVIDIA H100 SXM FP16 tensor throughput per GPU
        memory_bandwidth=3.35e12,
        dtype_bytes=2.0,
        activation_io_multiplier=12.0,
        PCIe_bandwidth=120e9,
        HBM_size=8.59e10,
        gpu_count=4,
    ),
    "H100_80GB_FP8_TP1": HardwareConfig(
        flops_per_second=3.958e15,
        memory_bandwidth=3.35e12,
        dtype_bytes=1.0,
        activation_io_multiplier=12.0,
        PCIe_bandwidth=120e9,
        HBM_size=8.59e10,
        gpu_count=1,
    ),
}

MODEL_PRESETS: Dict[str, ModelConfig] = {
    "7B": ModelConfig(hidden_size=4096, num_layers=32, expansion_ratio=4.0, model_size=7e9),
    DEFAULT_MODEL_NAME: ModelConfig(hidden_size=8192, num_layers=80, expansion_ratio=4.0, model_size=80e9),
    DEFAULT_DECODE_MODEL_NAME: ModelConfig(hidden_size=4096, num_layers=64, expansion_ratio=4.0, model_size=64e9),
    "llama33_70B": ModelConfig(hidden_size=8192, num_layers=80, expansion_ratio=3.5, model_size=70e9),
    "llama31_8B": ModelConfig(hidden_size=4096, num_layers=32, expansion_ratio=3.5, model_size=8e9),
}

WORKLOAD_PRESETS: Dict[str, WorkloadConfig] = {
    "default": WorkloadConfig(total_prompt_tokens=20_000.0),
}

S_L_GRID_SETTINGS = {
    "decode": {
        "context_min": 4.0,
        "context_samples": 400,
        "batch_range": (1.0, 5000, 600),
        "surface_batch_range": (1.0, 1001.0, 50.0),
        "surface_past_length_range": (32.0, 40000.0, 64.0),
    },
    "prefill": {
        "context_linspace": (1.0, 40_000.0, 400),
        "sample_batch_sizes": (1, 8, 16, 32, 64, 96),
        "batch_linspace": (1.0, 512.0, 50),
        "sample_context_lengths": (512, 2048, 8192, 20_000),
        "summary_points": (
            {"S": 1, "L": 1024},
            {"S": 8, "L": 1024},
            {"S": 32, "L": 2048},
            {"S": 64, "L": 8192},
            {"S": 128, "L": 20_000},
        ),
        "surface_batch_sizes": (1, 2, 4, 8, 16, 32, 64, 128),
        "surface_context_range": (128, 8192, 128),
    },
}


def get_hardware_config(name: str = DEFAULT_HARDWARE_NAME) -> HardwareConfig:
    """Return a hardware preset by name."""

    try:
        return HARDWARE_PRESETS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown hardware preset: {name}") from exc


def     get_model_config(name: str = DEFAULT_MODEL_NAME) -> ModelConfig:
    """Return a model preset by name."""

    try:
        return MODEL_PRESETS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown model preset: {name}") from exc


def get_workload_config(name: str = "default") -> WorkloadConfig:
    """Return a workload configuration by name."""

    try:
        return WORKLOAD_PRESETS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown workload preset: {name}") from exc


def decode_surface_grids() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return the (S, L) ranges for decode surface plots."""

    decode_settings = S_L_GRID_SETTINGS["decode"]
    return (
        decode_settings["surface_batch_range"],
        decode_settings["surface_past_length_range"],
    )


def prefill_surface_grids() -> Tuple[Tuple[int, ...], Tuple[int, int, int]]:
    """Return the (S, L) grids for prefill surface plots."""

    prefill_settings = S_L_GRID_SETTINGS["prefill"]
    return (
        prefill_settings["surface_batch_sizes"],
        prefill_settings["surface_context_range"],
    )
