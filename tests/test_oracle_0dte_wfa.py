from datetime import date, time, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

import oracle_0dte_wfa as oracle


def _candidate_frame(candidates: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(candidates)
    columns = sorted(frame.columns.tolist())
    return frame.reindex(columns=columns)


def _result_frame(results: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    columns = sorted(frame.columns.tolist())
    return frame.reindex(columns=columns)


def _build_chain(underlying_price: float) -> pd.DataFrame:
    rows: list[dict[str, float | bool]] = []
    for strike in [100, 95, 90, 85, 80, 75, 105, 110, 115, 120, 125]:
        distance = abs(strike - underlying_price)
        mid = max(0.1, 12.0 - 0.45 * distance)
        bid = mid - 0.10
        ask = mid + 0.10
        rows.append(
            {
                "strike": float(strike),
                "is_put": True,
                "mid": float(mid),
                "bid": float(bid),
                "ask": float(ask),
            }
        )
        call_mid = mid - 0.05
        rows.append(
            {
                "strike": float(strike),
                "is_put": False,
                "mid": float(call_mid),
                "bid": float(call_mid - 0.10),
                "ask": float(call_mid + 0.10),
            }
        )
    rows.append({"strike": 95.0, "is_put": True, "mid": 8.1, "bid": 8.0, "ask": 8.2})
    rows.append({"strike": 105.0, "is_put": False, "mid": 9.1, "bid": 9.0, "ask": 9.2})
    return pd.DataFrame(rows)


def _build_static_data() -> tuple[oracle.BacktestStaticData, pd.Timestamp]:
    minute_value = time(9, 31)
    minute_sec = minute_value.hour * 3600 + minute_value.minute * 60
    trading_days = [date(2025, 1, 1) + timedelta(days=offset) for offset in range(141)]
    trading_days_ord = np.array(
        [trading_day.toordinal() for trading_day in trading_days],
        dtype=np.int32,
    )

    hist_und_values = np.linspace(5900.0, 6040.0, num=len(trading_days_ord)).astype(
        np.float32
    )
    hist_exp_values = (
        hist_und_values + np.sin(np.arange(len(trading_days_ord), dtype=np.float32)) * 9
    ).astype(np.float32)

    hist_und: dict[tuple[int, int], float] = {}
    hist_exp: dict[tuple[int, int], float] = {}
    for trading_day_ord, und_value, exp_value in zip(
        trading_days_ord, hist_und_values, hist_exp_values, strict=False
    ):
        key = (int(trading_day_ord), minute_sec)
        hist_und[key] = float(und_value)
        hist_exp[key] = float(exp_value)

    static_data = oracle.BacktestStaticData(
        exp_price=pd.Series(
            hist_exp_values,
            index=pd.Index(trading_days, dtype=object),
            dtype=np.float32,
        ),
        trading_days=trading_days,
        trading_days_ord=trading_days_ord,
        test_days=trading_days[-5:],
        minute_secs=np.array([minute_sec], dtype=np.int32),
        day_ord_to_index={
            int(trading_day_ord): index
            for index, trading_day_ord in enumerate(trading_days_ord)
        },
        minute_sec_to_index={minute_sec: 0},
        hist_und=hist_und,
        hist_exp=hist_exp,
        hist_und_tensor=torch.tensor(
            hist_und_values.reshape(-1, 1),
            device=oracle.DEVICE,
            dtype=torch.float32,
        ),
        hist_exp_tensor=torch.tensor(
            hist_exp_values,
            device=oracle.DEVICE,
            dtype=torch.float32,
        ),
        holidays_count=0,
    )
    current_dt = cast(
        pd.Timestamp,
        pd.Timestamp(trading_days[-1]).replace(
            hour=minute_value.hour,
            minute=minute_value.minute,
        ),
    )
    return static_data, current_dt


def _reference_summary(df_cal: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_cal.groupby(["minute", "min_wr_thresh", "min_ror_thresh"], sort=False)
        .agg(
            n_trades=("ror", "count"),
            predicted_win_rate=("predicted_wr", "mean"),
            win_rate=("actual_win", "mean"),
            avg_ror=("ror", "mean"),
            predicted_ror=("predicted_ror", "mean"),
            avg_win=(
                "ror",
                lambda series: (
                    float(series[series > 0].mean()) if (series > 0).any() else 0.0
                ),
            ),
            avg_loss=(
                "ror",
                lambda series: (
                    float(series[series < 0].mean()) if (series < 0).any() else 0.0
                ),
            ),
            avg_pnl_pts=("pnl_pts", "mean"),
            avg_credit_pts=("credit_pts", "mean"),
            avg_risk_pts=("risk_pts", "mean"),
            avg_pnl_dollars=("pnl_dollars", "mean"),
            avg_credit_dollars=("credit_dollars", "mean"),
            avg_risk_dollars=("risk_dollars", "mean"),
            avg_hist_obs=("n_hist", "mean"),
            sharpe=("ror", oracle.safe_sharpe),
            sortino=("ror", oracle.safe_sortino),
        )
        .reset_index()
    )
    return summary


def test_generate_candidates_batch_no_calls_in_chain() -> None:
    """Regression: empty call_strikes must not crash _gen_ib_vec (IndexError)."""
    # Chain with only put quotes — no calls at all
    rows: list[dict[str, float | bool]] = []
    for strike in [95.0, 90.0, 85.0]:
        rows.append(
            {"strike": strike, "is_put": True, "mid": 5.0, "bid": 4.9, "ask": 5.1}
        )
    chain = pd.DataFrame(rows)

    # Should not raise IndexError
    result = oracle.generate_candidates_batch(100.0, chain)
    dict_result = oracle.generate_real_candidates(100.0, chain)

    # Both should handle gracefully (may return empty or puts-only)
    if result is not None:
        assert len(result.credit_f64) > 0
    assert isinstance(dict_result, list)


def test_generate_candidates_batch_matches_dict_output() -> None:
    """Verify vectorized generate_candidates_batch matches dict-based generate_real_candidates."""
    chain = _build_chain(underlying_price=100.0)

    dict_candidates = oracle.generate_real_candidates(100.0, chain, max_cand=5000)
    batch = oracle.generate_candidates_batch(100.0, chain, max_cand=5000)

    assert batch is not None
    assert len(dict_candidates) == len(batch.credit_f64)

    # Compare via the subset of columns both representations share
    common_cols = ["type", "credit", "risk", "width", "bid", "ask"]
    dict_rows = [{k: c[k] for k in common_cols} for c in dict_candidates]
    batch_rows = [
        {k: batch.get_candidate_dict(i)[k] for k in ["type", "credit", "risk", "width"]}
        | {"bid": float(batch.bid_f64[i]), "ask": float(batch.ask_f64[i])}
        for i in range(len(batch.credit_f64))
    ]

    assert oracle.dataframes_match_unordered(
        _candidate_frame(dict_rows),
        _candidate_frame(batch_rows),
    )


def test_candidates_have_bid_ask_with_real_chain() -> None:
    """Verify candidates include bid/ask that reflect the position's net bid-ask spread."""
    chain = _build_chain(underlying_price=100.0)
    candidates = oracle.generate_real_candidates(100.0, chain, max_cand=5000)
    assert len(candidates) > 0

    for c in candidates:
        assert "bid" in c, f"Missing 'bid' in candidate: {c}"
        assert "ask" in c, f"Missing 'ask' in candidate: {c}"
        # ask >= bid (position ask should be >= position bid)
        assert (
            c["ask"] >= c["bid"]
        ), f"ask < bid for {c['type']}: ask={c['ask']}, bid={c['bid']}"
        # Mid-credit should be between bid and ask
        assert (
            c["bid"] <= c["credit"] + 1e-9
        ), f"bid > credit for {c['type']}: bid={c['bid']}, credit={c['credit']}"
        assert (
            c["ask"] >= c["credit"] - 1e-9
        ), f"ask < credit for {c['type']}: ask={c['ask']}, credit={c['credit']}"


def test_oracle_batch_stats_returns_valid_results() -> None:
    """Verify oracle_batch_stats returns correctly-structured results with expected keys."""
    static_data, current_dt = _build_static_data()
    chain = _build_chain(underlying_price=103.0)
    candidates = oracle.generate_real_candidates(103.0, chain, max_cand=5000)

    candidate_by_type: dict[str, dict] = {}
    for candidate in candidates:
        candidate_by_type.setdefault(candidate["type"], candidate)

    selected_candidates = [
        candidate_by_type["pcs"],
        candidate_by_type["ccs"],
        candidate_by_type["ic"],
        candidate_by_type["ib"],
    ]

    results = oracle.oracle_batch_stats(
        current_dt=current_dt,
        candidates=selected_candidates,
        static_data=static_data,
        lookback=oracle.LOOKBACK_DAYS,
    )

    assert len(results) == len(selected_candidates)
    for result in results:
        assert "win_rate" in result
        assert "avg_ror" in result
        assert "n_hist" in result
        assert 0.0 <= result["win_rate"] <= 1.0
        assert result["n_hist"] > 0


def test_summarize_results_matches_reference_aggregation() -> None:
    df_cal = pd.DataFrame(
        [
            {
                "date": date(2026, 3, 2),
                "minute": "09:31",
                "min_wr_thresh": 0.50,
                "min_ror_thresh": 0.0,
                "predicted_wr": 0.60,
                "actual_win": 1,
                "ror": 0.12,
                "strategy_type": "pcs",
                "width": 5.0,
                "predicted_ror": 0.10,
                "n_hist": 130,
                "pnl_pts": 0.6,
                "credit_pts": 1.2,
                "risk_pts": 4.4,
                "pnl_dollars": 60.0,
                "credit_dollars": 120.0,
                "risk_dollars": 440.0,
            },
            {
                "date": date(2026, 3, 2),
                "minute": "09:31",
                "min_wr_thresh": 0.50,
                "min_ror_thresh": 0.0,
                "predicted_wr": 0.55,
                "actual_win": 0,
                "ror": -0.08,
                "strategy_type": "pcs",
                "width": 5.0,
                "predicted_ror": 0.07,
                "n_hist": 128,
                "pnl_pts": -0.4,
                "credit_pts": 1.1,
                "risk_pts": 4.5,
                "pnl_dollars": -40.0,
                "credit_dollars": 110.0,
                "risk_dollars": 450.0,
            },
            {
                "date": date(2026, 3, 3),
                "minute": "09:31",
                "min_wr_thresh": 0.50,
                "min_ror_thresh": 0.0,
                "predicted_wr": 0.65,
                "actual_win": 1,
                "ror": 0.05,
                "strategy_type": "ccs",
                "width": 10.0,
                "predicted_ror": 0.06,
                "n_hist": 140,
                "pnl_pts": 0.5,
                "credit_pts": 1.5,
                "risk_pts": 8.5,
                "pnl_dollars": 50.0,
                "credit_dollars": 150.0,
                "risk_dollars": 850.0,
            },
            {
                "date": date(2026, 3, 2),
                "minute": "09:32",
                "min_wr_thresh": 0.70,
                "min_ror_thresh": 0.0,
                "predicted_wr": 0.72,
                "actual_win": 0,
                "ror": -0.04,
                "strategy_type": "ic",
                "width": 5.0,
                "predicted_ror": 0.03,
                "n_hist": 135,
                "pnl_pts": -0.2,
                "credit_pts": 1.0,
                "risk_pts": 4.0,
                "pnl_dollars": -20.0,
                "credit_dollars": 100.0,
                "risk_dollars": 400.0,
            },
            {
                "date": date(2026, 3, 3),
                "minute": "09:32",
                "min_wr_thresh": 0.70,
                "min_ror_thresh": 0.0,
                "predicted_wr": 0.74,
                "actual_win": 1,
                "ror": 0.09,
                "strategy_type": "ic",
                "width": 5.0,
                "predicted_ror": 0.08,
                "n_hist": 137,
                "pnl_pts": 0.45,
                "credit_pts": 1.1,
                "risk_pts": 3.9,
                "pnl_dollars": 45.0,
                "credit_dollars": 110.0,
                "risk_dollars": 390.0,
            },
            {
                "date": date(2026, 3, 4),
                "minute": "09:32",
                "min_wr_thresh": 0.70,
                "min_ror_thresh": 0.0,
                "predicted_wr": 0.76,
                "actual_win": 1,
                "ror": 0.02,
                "strategy_type": "ib",
                "width": 5.0,
                "predicted_ror": 0.04,
                "n_hist": 138,
                "pnl_pts": 0.1,
                "credit_pts": 1.0,
                "risk_pts": 4.0,
                "pnl_dollars": 10.0,
                "credit_dollars": 100.0,
                "risk_dollars": 400.0,
            },
        ]
    )

    expected = _reference_summary(df_cal)
    actual = oracle.summarize_results(df_cal)

    assert oracle.dataframes_match_unordered(expected, actual)


def test_save_and_load_results_csv_round_trip(tmp_path: Path) -> None:
    df_cal = pd.DataFrame(
        [
            {
                "date": date(2026, 3, 2),
                "minute": "09:31",
                "min_wr_thresh": 0.50,
                "min_ror_thresh": 0.0,
                "predicted_wr": 0.62,
                "actual_win": 1,
                "ror": 0.11,
                "strategy_type": "pcs",
                "width": 5.0,
                "predicted_ror": 0.09,
                "n_hist": 131,
                "pnl_pts": 0.55,
                "credit_pts": 1.15,
                "risk_pts": 4.45,
                "pnl_dollars": 55.0,
                "credit_dollars": 115.0,
                "risk_dollars": 445.0,
            }
        ]
    )
    res_summary = oracle.summarize_results(df_cal)

    oracle.save_results_to_csv(
        df_cal,
        res_summary,
        label="roundtrip",
        test_start=date(2026, 3, 2),
        output_dir=tmp_path,
    )
    loaded_df_cal, loaded_res_summary = oracle.load_results_from_csv(
        label="roundtrip",
        test_start=date(2026, 3, 2),
        output_dir=tmp_path,
    )

    assert oracle.dataframes_match_unordered(df_cal, loaded_df_cal)
    assert oracle.dataframes_match_unordered(res_summary, loaded_res_summary)
