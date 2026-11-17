"""Apply exported strategy predicates to a raw trade ledger without optimization."""

from __future__ import annotations

from pathlib import Path

import torch

from ._conditions import load_conditions
from ._data import load_strategy_csv
from ._date_window import select_strategy_dates
from ._evaluation import StrategyEvaluator
from ._output import save_strategy, winning_trades
from ._preparation import prepare_strategy
from .optimizer import MarketDataSource, _load_market


def apply_strategy_conditions(
    raw_csv: Path | str,
    conditions_yaml: Path | str,
    *,
    output_csv: Path | str | None = None,
    market_data_source: MarketDataSource | None = None,
    device: torch.device | str | None = None,
    observation_batch_size: int | None = None,
) -> Path:
    """Recreate a strategy CSV using a saved entry filter and/or exit predicate.

    Reuse the YAML's frozen feature definitions, date window, indicator lookback,
    and whole-combination exit slippage. Market data is loaded in the same way as
    optimization, including Oracle/Yahoo/event caches unless an offline source
    is supplied. Identical input/market data reproduces the original export.

    The default destination is ``<conditions stem>_recreated.csv`` beside the
    YAML, omitting its ``.conditions`` suffix. Both inputs remain untouched.
    Return the written CSV path; no evolution, metrics, allocation, or plots run.
    """
    artifact = Path(conditions_yaml)
    saved = load_conditions(artifact)
    data = select_strategy_dates(
        load_strategy_csv(raw_csv), saved.start_date, saved.end_date
    )
    destination = (
        Path(output_csv)
        if output_csv is not None
        else artifact.with_name(
            artifact.stem.removesuffix(".conditions") + "_recreated.csv"
        )
    )
    if destination.resolve() in (data.path.resolve(), artifact.resolve()):
        raise ValueError("output_csv must differ from the raw CSV and condition YAML")
    selected_device = torch.device(
        device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )
    if selected_device.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    market = _load_market(
        data,
        market_data_source,
        saved.lookback_days,
        any(feature.needs_daily_history for feature in saved.features),
    )
    prepared = prepare_strategy(
        data, market, saved.features, saved.mode, saved.exit_slippage
    )
    evaluator = StrategyEvaluator(
        prepared.tensors.to(selected_device),
        saved.features,
        saved.mode,
        observation_batch_size,
    )
    evaluation = evaluator.evaluate(
        saved.bounds.to(selected_device)[None], saved.active.to(selected_device)[None]
    )
    return save_strategy(winning_trades(prepared, evaluation), destination, data.path)
