# llm-guesstimator

Analytical tooling for estimating large language model (LLM) inference latency and throughput. The project derives closed-form formulas for prefill (TTFT) and decode (ITL) phases, compares their compute vs. memory limits on modern GPUs, and validates the estimators against real benchmark runs.

## Notebook Guide

### prefill_estimations.ipynb
Derives FLOP counts and memory traffic for the prefill stage, assuming a 2·m·n·p convention for matmul cost. Sweeps sequence length `L` and batch size `S` to predict compute time, memory time, and their crossover points on hardware such as the A100-40GB.

### decode_estimations.ipynb
Builds per-token decode estimates with KV-cache reuse. Walks through the complexity model, keeps the `P = S·L` token budget constant, and visualizes how compute and memory ceilings shape latency as prompts grow.

### prefill_decode_combined.ipynb
Consolidates the corrected prefill and decode formulas into a single playground. Compares arithmetic intensity with machine balance, surfaces when each phase becomes compute- or memory-bound, and plots the max of the two across practical ranges of `S` and `L`.

### benchmark_validation.ipynb
Loads measured TTFT and ITL results (see `evaluations_results/`), matches them with configuration details, and recomputes the analytical expectations. The notebook produces per-scenario error summaries and overlay plots so the analytical model can be tuned against real hardware traces.

## Data Inputs
- `evaluations_results/`: processed latency measurements used by the validation notebook.
- `tested_benchmarks/`: raw benchmark exports that feed into the processed evaluation files.
- `utils/`: helper modules shared across the notebooks (loading, plotting, math helpers).

## Results
_TODO: Summarize the primary takeaways from `benchmark_validation.ipynb` plots (e.g., estimator error bands, bottleneck shifts, GPU-specific observations)._ 
