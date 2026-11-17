import os

os.environ["IFERA_DB_USER"] = "iferaro"
os.environ["IFERA_DB_PASSWORD"] = "iferaro"
os.environ["IFERA_DSN"] = "IFERA1"
os.environ["TRITON_DEFAULT_FP_FUSION"] = "0"

from datetime import date
import torch

from ifera.strategy_optimization import (
    optimize_strategy_from_csv,
    return_over_add,
    annualized_return,
    with_scarcity_penalty
)

torch._dynamo.config.capture_scalar_outputs = True

metric = with_scarcity_penalty(return_over_add, 0.1, 0.4)
_cuda_metric = torch.compile(metric, dynamic=True)

result = optimize_strategy_from_csv(
    "pf_005_exp",
    "AftPC2_raw",
    mode="combined",  # entry_filter, early_exit, or combined
    metric=_cuda_metric,
    optimizer_parameters={
        "max_conjunctions": 8,
        "max_population": 2**12,
        "generations": 10000,
        "crossover_rate": 1.0,
        "fine_grained_crossover_probability": 1.0,
        "survival_rate": 0.5,
        "boolean_complexity_penalty": 0.001,
        "continuous_complexity_penalty": 0.002,
    },
    exit_slippage=0.05,
    start_date=date(2023, 1, 1),
    # end_date=date(2026, 11, 12),
    # features_file="my_features.yaml",
    # output_csv="optimized.csv",
)

print(result.conditions)
print(result.statistics)
