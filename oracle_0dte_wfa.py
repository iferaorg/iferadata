"""Standalone SPXW 0DTE walk-forward oracle script with legacy parity hooks."""

# pyright: reportReturnType=false, reportArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
# pyright: reportGeneralTypeIssues=false, reportRedeclaration=false
# pyright: reportIndexIssue=false
# pylint: disable=too-many-lines,missing-function-docstring
# pylint: disable=function-redefined,redefined-outer-name
# pylint: disable=too-many-branches,too-many-statements

import urllib.parse
import warnings
from dataclasses import dataclass
from datetime import date, time
from itertools import product
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sqlalchemy import create_engine, Engine
from tqdm.auto import tqdm

warnings.filterwarnings("default")

# ========================== ORACLE CONNECTION ==========================
DB_CONFIG = {
    "user": "iferaro",  # ← CHANGE
    # Local read-only Oracle credential used by this standalone script.
    "password": "iferaro",  # nosec B105
    "dsn": "IFERA1",  # ← CHANGE
}

user = urllib.parse.quote_plus(DB_CONFIG["user"])
password = urllib.parse.quote_plus(DB_CONFIG["password"])
dsn = urllib.parse.quote_plus(DB_CONFIG["dsn"])

# ========================== USER CONFIG ==========================
CONTRACT_MULT = 100
MIN_OI = 100
MAX_BA_PCT = 0.20
MAX_OTM_PCT = 0.03
MAX_CANDIDATES_PER_MINUTE = 5000
SPREAD_WIDTHS = [5, 10, 15, 20, 25]

TEST_MINUTES = [
    time(h, m)
    for h, m in product(range(9, 16), range(0, 60, 1))
    if not (h == 9 and m < 31) and not (h == 15 and m > 50)
]
MIN_WR_THRESHOLDS = np.round(np.arange(0.0, 0.91, 0.05), 2)
MIN_ROR_THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.1), 1)

SLIPPAGE_PCT_OF_BASPREAD = 0.0
LOOKBACK_DAYS = 256
MIN_HISTORY_OBS = max(30, LOOKBACK_DAYS // 2)
FULL_TEST_START = date(2024, 7, 5)
SHORT_TEST_START = date(
    2026, 3, 2
)  # For faster performance / easier debugging with less data
TEST_START = FULL_TEST_START  # Change to FULL_TEST_START for production runs

# Oracle DATETIME is TIMESTAMP(3) without timezone, so keep all DB-facing
# timestamps naive and interpret them as already in the market's local clock.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class BacktestStaticData:
    """Pre-loaded day/minute data reused across walk-forward iterations."""

    exp_price: pd.Series
    trading_days: List[date]
    trading_days_ord: np.ndarray
    test_days: List[date]
    minute_secs: np.ndarray
    day_ord_to_index: Dict[int, int]
    minute_sec_to_index: Dict[int, int]
    hist_und: Dict[tuple, float]
    hist_exp: Dict[tuple, float]
    hist_und_tensor: torch.Tensor
    hist_exp_tensor: torch.Tensor
    holidays_count: int


# ========================== HELPERS ==========================
def get_timestamp(d: date, t: time) -> pd.Timestamp:
    """Return a DB-compatible timestamp.

    OPTION_SNAPSHOT_1M.DATETIME is TIMESTAMP(3) without timezone,
    so we deliberately use naive timestamps here.
    """
    return pd.Timestamp.combine(d, t)


def calculate_vertical_payoff(
    exp_price: float,
    short_strike: float,
    long_strike: float,
    direction: str,
) -> float:
    if direction == "put":
        return max(short_strike - exp_price, 0.0) - max(long_strike - exp_price, 0.0)
    if direction == "call":
        return max(exp_price - short_strike, 0.0) - max(exp_price - long_strike, 0.0)
    raise ValueError(f"Unknown direction: {direction}")


def safe_sharpe(x: pd.Series) -> float:
    if len(x) < 2:
        return 0.0
    std = x.std(ddof=1)
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return float(x.mean() / std * np.sqrt(252))


def safe_sortino(x: pd.Series) -> float:
    downside = x[x < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std(ddof=1)
    if not np.isfinite(downside_std) or downside_std <= 0:
        return 0.0
    return float(x.mean() / downside_std * np.sqrt(252))


def payoff_from_candidate(exp_price_value: float, candidate: Dict) -> float:
    ctype = candidate["type"]

    if ctype in ("pcs", "ccs"):
        return calculate_vertical_payoff(
            exp_price=exp_price_value,
            short_strike=candidate["short_strike"],
            long_strike=candidate["long_strike"],
            direction=candidate["direction"],
        )

    if ctype == "ic":
        put_payoff = calculate_vertical_payoff(
            exp_price=exp_price_value,
            short_strike=candidate["short_put_strike"],
            long_strike=candidate["long_put_strike"],
            direction="put",
        )
        call_payoff = calculate_vertical_payoff(
            exp_price=exp_price_value,
            short_strike=candidate["short_call_strike"],
            long_strike=candidate["long_call_strike"],
            direction="call",
        )
        return put_payoff + call_payoff

    if ctype == "ib":
        put_payoff = calculate_vertical_payoff(
            exp_price=exp_price_value,
            short_strike=candidate["short_put_strike"],
            long_strike=candidate["long_put_strike"],
            direction="put",
        )
        call_payoff = calculate_vertical_payoff(
            exp_price=exp_price_value,
            short_strike=candidate["short_call_strike"],
            long_strike=candidate["long_call_strike"],
            direction="call",
        )
        return put_payoff + call_payoff

    raise ValueError(f"Unknown candidate type: {ctype}")


def normalize_live_chain_frame(
    df: pd.DataFrame, include_datetime: bool = False
) -> pd.DataFrame:
    """Apply the same defensive cleanup to every live-chain load path."""
    if df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    normalized.columns = [col.lower() for col in normalized.columns]
    if "datetime" in normalized.columns:
        normalized["datetime"] = pd.to_datetime(normalized["datetime"])
    normalized["is_put"] = normalized["right"].eq("P")

    normalized = normalized.dropna(
        subset=["strike", "mid", "oi", "bid", "ask", "und_price"]
    ).copy()
    normalized = normalized[
        (normalized["mid"] > 0) & (normalized["ask"] >= normalized["bid"])
    ].copy()

    if normalized.empty:
        return pd.DataFrame()

    columns = ["strike", "is_put", "mid", "oi", "bid", "ask", "und_price"]
    if include_datetime:
        columns = ["datetime", *columns]
    return normalized[columns].copy()


def load_live_spx_chain_day(
    engine: Engine, trade_date: date
) -> Dict[pd.Timestamp, pd.DataFrame]:
    day_start = get_timestamp(trade_date, time(0, 0))
    day_end = day_start + pd.Timedelta(days=1)
    query = """
    SELECT
        DATETIME,
        STRIKE,
        RIGHT,
        MIDPOINT AS mid,
        OPEN_INTEREST AS oi,
        BID,
        ASK,
        UNDERLYING_PRICE AS und_price
    FROM OPTION_SNAPSHOT_1M
    WHERE SYMBOL = 'SPXW'
      AND DATETIME >= :day_start
      AND DATETIME < :day_end
      AND EXP_DATE = :trade_date
      AND MIDPOINT > 0.05
      AND OPEN_INTEREST >= :min_oi
      AND BID IS NOT NULL
      AND ASK IS NOT NULL
      AND BID > 0
      AND ASK > 0
      AND ASK >= BID
      AND ((ASK - BID) / MIDPOINT <= :max_ba_pct OR ASK - BID <= 0.2)
    ORDER BY DATETIME, STRIKE, RIGHT
    """
    df = pd.read_sql(
        query,
        engine,
        params={
            "day_start": day_start,
            "day_end": day_end,
            "trade_date": trade_date,
            "min_oi": MIN_OI,
            "max_ba_pct": MAX_BA_PCT,
        },
    )
    normalized = normalize_live_chain_frame(df, include_datetime=True)
    if normalized.empty:
        return {}

    grouped: Dict[pd.Timestamp, pd.DataFrame] = {}
    for entry_dt, minute_df in normalized.groupby("datetime", sort=False):
        grouped[pd.Timestamp(entry_dt)] = minute_df.drop(
            columns="datetime"
        ).reset_index(drop=True)

    return grouped


def load_static_backtest_data(engine: Engine) -> BacktestStaticData:
    print("Pre-loading underlying prices, settlement proxy, and holidays...")

    und_query = """
    SELECT DATETIME, AVG(UNDERLYING_PRICE) AS close
    FROM OPTION_SNAPSHOT_1M
    WHERE SYMBOL = 'SPXW'
    GROUP BY DATETIME
    ORDER BY DATETIME
    """
    spx_minute = pd.read_sql(und_query, engine)
    spx_minute["datetime"] = pd.to_datetime(spx_minute["datetime"])
    spx_minute = spx_minute.set_index("datetime")["close"].sort_index()

    settle_query = """
    SELECT
        TRUNC(DATETIME) AS exp_date,
        MAX(UNDERLYING_PRICE) KEEP (DENSE_RANK FIRST ORDER BY DATETIME DESC) AS settlement
    FROM OPTION_SNAPSHOT_1M
    WHERE SYMBOL = 'SPXW'
    GROUP BY TRUNC(DATETIME)
    """
    exp_df = pd.read_sql(settle_query, engine)
    exp_df["exp_date"] = pd.to_datetime(exp_df["exp_date"]).dt.date
    exp_price = exp_df.set_index("exp_date")["settlement"].sort_index()

    holiday_query = "SELECT T_DATE FROM TDC_LIST_HOLIDAY"
    holidays = set(
        pd.to_datetime(pd.read_sql(holiday_query, engine)["t_date"]).dt.date.tolist()
    )

    valid_days = [d for d in exp_price.index if d not in holidays]
    exp_price = exp_price[exp_price.index.isin(valid_days)]
    trading_days = sorted(valid_days)
    trading_days_ord = np.array([d.toordinal() for d in trading_days], dtype=np.int32)
    test_days = [d for d in trading_days if d >= TEST_START]

    minute_secs = np.array(
        [t.hour * 3600 + t.minute * 60 for t in TEST_MINUTES], dtype=np.int32
    )

    hist_und: Dict[tuple, float] = {}
    hist_exp: Dict[tuple, float] = {}
    hist_und_matrix = np.full(
        (len(trading_days_ord), len(minute_secs)), np.nan, dtype=np.float32
    )
    hist_exp_values = np.full(len(trading_days_ord), np.nan, dtype=np.float32)

    for i, d_ord in enumerate(trading_days_ord):
        current_date = trading_days[i]
        exp_val = exp_price.get(current_date)
        if pd.isna(exp_val):
            continue
        hist_exp_values[i] = float(exp_val)

        for minute_index, m_sec in enumerate(minute_secs):
            ts = pd.Timestamp.combine(
                date.fromordinal(int(d_ord)),
                time(int(m_sec // 3600), int((m_sec % 3600) // 60)),
            )
            und_val = spx_minute.get(ts, np.nan)
            if pd.isna(und_val):
                continue

            hist_und_matrix[i, minute_index] = float(und_val)
            key = (int(d_ord), int(m_sec))
            hist_und[key] = float(und_val)
            hist_exp[key] = float(exp_val)

    print(f"Loaded {len(holidays)} holidays → excluded.")
    print(f"Final trading days: {len(trading_days):,}")
    print(f"Test trading days : {len(test_days):,}")
    print(f"Historical minute keys cached: {len(hist_und):,}")

    return BacktestStaticData(
        exp_price=exp_price,
        trading_days=trading_days,
        trading_days_ord=trading_days_ord,
        test_days=test_days,
        minute_secs=minute_secs,
        day_ord_to_index={
            int(trading_day_ord): index
            for index, trading_day_ord in enumerate(trading_days_ord)
        },
        minute_sec_to_index={
            int(minute_sec): index for index, minute_sec in enumerate(minute_secs)
        },
        hist_und=hist_und,
        hist_exp=hist_exp,
        hist_und_tensor=torch.tensor(
            hist_und_matrix, device=DEVICE, dtype=torch.float32
        ),
        hist_exp_tensor=torch.tensor(
            hist_exp_values, device=DEVICE, dtype=torch.float32
        ),
        holidays_count=len(holidays),
    )


def dataframes_match_unordered(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    if left.empty and right.empty:
        return True

    sort_columns = list(left.columns)
    left_sorted = left.sort_values(sort_columns, kind="mergesort").reset_index(
        drop=True
    )
    right_sorted = right.sort_values(sort_columns, kind="mergesort").reset_index(
        drop=True
    )

    for column in sort_columns:
        left_col = left_sorted[column]
        right_col = right_sorted[column]
        if pd.api.types.is_numeric_dtype(left_col) and pd.api.types.is_numeric_dtype(
            right_col
        ):
            if not np.allclose(
                left_col.to_numpy(),
                right_col.to_numpy(),
                equal_nan=True,
                rtol=rtol,
                atol=atol,
            ):
                return False
            continue

        if not left_col.equals(right_col):
            return False

    return True


def get_results_csv_paths(
    label: str = "old", test_start: date = TEST_START, output_dir: Path | str = "."
) -> tuple[Path, Path]:
    base_dir = Path(output_dir)
    suffix = test_start.isoformat()
    return (
        base_dir / f"oracle_0dte_wfa_{label}_df_cal_{suffix}.csv",
        base_dir / f"oracle_0dte_wfa_{label}_res_summary_{suffix}.csv",
    )


def save_results_to_csv(
    df_cal: pd.DataFrame,
    res_summary: pd.DataFrame,
    *,
    label: str = "old",
    test_start: date = TEST_START,
    output_dir: Path | str = ".",
) -> tuple[Path, Path]:
    df_cal_path, res_summary_path = get_results_csv_paths(
        label=label, test_start=test_start, output_dir=output_dir
    )
    df_cal.to_csv(df_cal_path, index=False)
    res_summary.to_csv(res_summary_path, index=False)
    return df_cal_path, res_summary_path


def load_results_from_csv(
    *,
    label: str = "old",
    test_start: date = TEST_START,
    output_dir: Path | str = ".",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_cal_path, res_summary_path = get_results_csv_paths(
        label=label, test_start=test_start, output_dir=output_dir
    )
    df_cal = pd.read_csv(df_cal_path, parse_dates=["date"])
    df_cal["date"] = df_cal["date"].dt.date
    res_summary = pd.read_csv(res_summary_path)
    return df_cal, res_summary


def _prepare_option_quotes(
    strikes_all: np.ndarray,
    mids_all: np.ndarray,
    is_put_all: np.ndarray,
    und_price: float,
    *,
    is_put: bool,
    bids_all: np.ndarray | None = None,
    asks_all: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_mask = is_put_all if is_put else ~is_put_all
    strikes = strikes_all[selected_mask]
    if strikes.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    mids = mids_all[selected_mask]
    bids = bids_all[selected_mask] if bids_all is not None else mids.copy()
    asks = asks_all[selected_mask] if asks_all is not None else mids.copy()
    order = np.argsort(strikes, kind="mergesort")
    strikes = strikes[order]
    mids = mids[order]
    bids = bids[order]
    asks = asks[order]

    keep_mask = np.empty(strikes.shape[0], dtype=bool)
    keep_mask[0] = True
    keep_mask[1:] = strikes[1:] != strikes[:-1]
    strikes = strikes[keep_mask]
    mids = mids[keep_mask]
    bids = bids[keep_mask]
    asks = asks[keep_mask]

    if is_put:
        otm_pct = (und_price - strikes) / und_price
    else:
        otm_pct = (strikes - und_price) / und_price

    return strikes, mids, otm_pct, bids, asks


def generate_real_candidates(
    und_price: float,
    chain: pd.DataFrame,
    max_cand: int = MAX_CANDIDATES_PER_MINUTE,
) -> List[Dict]:
    if max_cand <= 0 or chain.empty or not np.isfinite(und_price) or und_price <= 0:
        return []

    strikes_all = chain["strike"].to_numpy(dtype=np.float64, copy=False)
    mids_all = chain["mid"].to_numpy(dtype=np.float64, copy=False)
    is_put_all = chain["is_put"].to_numpy(dtype=bool, copy=False)
    has_ba = "bid" in chain.columns and "ask" in chain.columns
    bids_all = chain["bid"].to_numpy(dtype=np.float64, copy=False) if has_ba else None
    asks_all = chain["ask"].to_numpy(dtype=np.float64, copy=False) if has_ba else None

    put_strikes, put_mids, put_otm_pct, put_bids, put_asks = _prepare_option_quotes(
        strikes_all,
        mids_all,
        is_put_all,
        und_price,
        is_put=True,
        bids_all=bids_all,
        asks_all=asks_all,
    )
    call_strikes, call_mids, call_otm_pct, call_bids, call_asks = (
        _prepare_option_quotes(
            strikes_all,
            mids_all,
            is_put_all,
            und_price,
            is_put=False,
            bids_all=bids_all,
            asks_all=asks_all,
        )
    )

    put_quotes = {
        float(strike): (float(mid), float(otm_pct), float(bid), float(ask))
        for strike, mid, otm_pct, bid, ask in zip(
            put_strikes, put_mids, put_otm_pct, put_bids, put_asks, strict=False
        )
    }
    call_quotes = {
        float(strike): (float(mid), float(otm_pct), float(bid), float(ask))
        for strike, mid, otm_pct, bid, ask in zip(
            call_strikes, call_mids, call_otm_pct, call_bids, call_asks, strict=False
        )
    }

    candidates: List[Dict] = []
    pcs_by_width: Dict[int, List[Dict]] = {width: [] for width in SPREAD_WIDTHS}
    ccs_by_width: Dict[int, List[Dict]] = {width: [] for width in SPREAD_WIDTHS}

    def append_candidate(candidate: Dict, bucket: List[Dict] | None = None) -> bool:
        if bucket is not None:
            bucket.append(candidate)
        candidates.append(candidate)
        return len(candidates) >= max_cand

    for short_strike, (
        short_mid,
        short_otm,
        short_bid,
        short_ask,
    ) in put_quotes.items():
        if short_otm < 0 or short_otm > MAX_OTM_PCT:
            continue

        for width in SPREAD_WIDTHS:
            long_strike = short_strike - width
            long_quote = put_quotes.get(long_strike)
            if long_quote is None:
                continue

            credit = short_mid - long_quote[0]
            if credit <= 0:
                continue

            candidate = {
                "type": "pcs",
                "direction": "put",
                "short_strike": short_strike,
                "long_strike": long_strike,
                "short_otm": short_otm,
                "width": float(width),
                "credit": float(credit),
                "bid": float(short_bid - long_quote[3]),
                "ask": float(short_ask - long_quote[2]),
            }
            if append_candidate(candidate, pcs_by_width[width]):
                break

        if len(candidates) >= max_cand:
            break

    if len(candidates) < max_cand:
        for short_strike, (
            short_mid,
            short_otm,
            short_bid,
            short_ask,
        ) in call_quotes.items():
            if short_otm < 0 or short_otm > MAX_OTM_PCT:
                continue

            for width in SPREAD_WIDTHS:
                long_strike = short_strike + width
                long_quote = call_quotes.get(long_strike)
                if long_quote is None:
                    continue

                credit = short_mid - long_quote[0]
                if credit <= 0:
                    continue

                candidate = {
                    "type": "ccs",
                    "direction": "call",
                    "short_strike": short_strike,
                    "long_strike": long_strike,
                    "short_otm": short_otm,
                    "width": float(width),
                    "credit": float(credit),
                    "bid": float(short_bid - long_quote[3]),
                    "ask": float(short_ask - long_quote[2]),
                }
                if append_candidate(candidate, ccs_by_width[width]):
                    break

            if len(candidates) >= max_cand:
                break

    if len(candidates) < max_cand:
        for width in SPREAD_WIDTHS:
            for put_candidate, call_candidate in product(
                pcs_by_width[width], ccs_by_width[width]
            ):
                if put_candidate["short_strike"] >= call_candidate["short_strike"]:
                    continue

                credit = put_candidate["credit"] + call_candidate["credit"]
                candidate = {
                    "type": "ic",
                    "short_put_strike": put_candidate["short_strike"],
                    "long_put_strike": put_candidate["long_strike"],
                    "short_call_strike": call_candidate["short_strike"],
                    "long_call_strike": call_candidate["long_strike"],
                    "put_short_otm": put_candidate["short_otm"],
                    "call_short_otm": call_candidate["short_otm"],
                    "width": put_candidate["width"],
                    "credit": float(credit),
                    "bid": float(put_candidate["bid"] + call_candidate["bid"]),
                    "ask": float(put_candidate["ask"] + call_candidate["ask"]),
                }
                if append_candidate(candidate):
                    break

            if len(candidates) >= max_cand:
                break

    if len(candidates) < max_cand:
        for center in np.unique(strikes_all):
            center_offset_pct = (float(center) - und_price) / und_price
            if abs(center_offset_pct) > MAX_OTM_PCT:
                continue

            short_put = put_quotes.get(float(center))
            short_call = call_quotes.get(float(center))
            if short_put is None or short_call is None:
                continue

            for width in SPREAD_WIDTHS:
                long_put = put_quotes.get(float(center - width))
                long_call = call_quotes.get(float(center + width))
                if long_put is None or long_call is None:
                    continue

                put_credit = short_put[0] - long_put[0]
                call_credit = short_call[0] - long_call[0]
                if put_credit <= 0 or call_credit <= 0:
                    continue

                credit = put_credit + call_credit
                # Put spread bid/ask
                p_bid = short_put[2] - long_put[3]
                p_ask = short_put[3] - long_put[2]
                # Call spread bid/ask
                c_bid = short_call[2] - long_call[3]
                c_ask = short_call[3] - long_call[2]
                candidate = {
                    "type": "ib",
                    "short_put_strike": float(center),
                    "short_call_strike": float(center),
                    "long_put_strike": float(center - width),
                    "long_call_strike": float(center + width),
                    "center_offset_pct": float(center_offset_pct),
                    "width": float(width),
                    "credit": float(credit),
                    "bid": float(p_bid + c_bid),
                    "ask": float(p_ask + c_ask),
                }
                if append_candidate(candidate):
                    break

            if len(candidates) >= max_cand:
                break

    for candidate in candidates:
        width = float(candidate["width"])
        candidate["risk"] = float(width - candidate["credit"])
        candidate["ror"] = (
            float(candidate["credit"] / candidate["risk"])
            if candidate["risk"] > 0
            else 0.0
        )

    return candidates


# ====================== COLUMNAR CANDIDATE BATCH ==============================

_TYPE_PCS = 0
_TYPE_CCS = 1
_TYPE_IC = 2
_TYPE_IB = 3
_TYPE_NAMES = ("pcs", "ccs", "ic", "ib")


class CandidateBatch:
    """Columnar storage for candidates – avoids millions of dict allocations."""

    __slots__ = (
        "n",
        "credit",
        "risk",
        "width",
        "param_a",
        "param_b",
        "type_id",
        "type_ranges",
        "strike_a",
        "strike_b",
        "strike_c",
        "strike_d",
        "credit_f64",
        "width_f64",
        "bid_f64",
        "ask_f64",
        "gpu_params",
    )

    def __init__(
        self,
        n: int,
        credit: np.ndarray,
        risk: np.ndarray,
        width: np.ndarray,
        param_a: np.ndarray,
        param_b: np.ndarray,
        type_id: np.ndarray,
        type_ranges: Dict[str, tuple[int, int]],
        strike_a: np.ndarray,
        strike_b: np.ndarray,
        strike_c: np.ndarray,
        strike_d: np.ndarray,
        credit_f64: np.ndarray,
        width_f64: np.ndarray,
        bid_f64: np.ndarray,
        ask_f64: np.ndarray,
    ) -> None:
        self.n = n
        self.credit = credit
        self.risk = risk
        self.width = width
        self.param_a = param_a
        self.param_b = param_b
        self.type_id = type_id
        self.type_ranges = type_ranges
        self.strike_a = strike_a
        self.strike_b = strike_b
        self.strike_c = strike_c
        self.strike_d = strike_d
        self.credit_f64 = credit_f64
        self.width_f64 = width_f64
        self.bid_f64 = bid_f64
        self.ask_f64 = ask_f64
        # Pre-built contiguous (n, 5) float32 array for single GPU transfer
        self.gpu_params = np.column_stack([credit, risk, param_a, param_b, width])

    def get_candidate_dict(self, idx: int) -> Dict:
        """Build candidate dict for a single winner (trade reporting)."""
        tid = int(self.type_id[idx])
        cr = float(self.credit_f64[idx])
        wd = float(self.width_f64[idx])
        d: Dict = {
            "type": _TYPE_NAMES[tid],
            "credit": cr,
            "risk": float(wd - cr),
            "width": wd,
        }
        if tid <= 1:
            d["short_strike"] = float(self.strike_a[idx])
            d["long_strike"] = float(self.strike_b[idx])
            d["direction"] = "put" if tid == 0 else "call"
        else:
            d["short_put_strike"] = float(self.strike_a[idx])
            d["long_put_strike"] = float(self.strike_b[idx])
            d["short_call_strike"] = float(self.strike_c[idx])
            d["long_call_strike"] = float(self.strike_d[idx])
        return d

    def payoffs_at(self, indices: np.ndarray, exp_price: float) -> np.ndarray:
        """Compute payoffs for multiple winner indices at once (vectorized)."""
        sa = self.strike_a[indices]
        sb = self.strike_b[indices]
        sc = self.strike_c[indices]
        sd = self.strike_d[indices]
        tids = self.type_id[indices]

        # PCS (tid=0): max(short - exp, 0) - max(long - exp, 0)
        pcs_mask = tids == _TYPE_PCS
        # CCS (tid=1): max(exp - short, 0) - max(exp - long, 0)
        ccs_mask = tids == _TYPE_CCS
        # IC/IB (tid=2,3): put_payoff + call_payoff
        ic_ib_mask = (tids == _TYPE_IC) | (tids == _TYPE_IB)

        payoff = np.zeros(len(indices), dtype=np.float64)
        if pcs_mask.any():
            payoff[pcs_mask] = np.maximum(sa[pcs_mask] - exp_price, 0.0) - np.maximum(
                sb[pcs_mask] - exp_price, 0.0
            )
        if ccs_mask.any():
            payoff[ccs_mask] = np.maximum(exp_price - sa[ccs_mask], 0.0) - np.maximum(
                exp_price - sb[ccs_mask], 0.0
            )
        if ic_ib_mask.any():
            pp = np.maximum(sa[ic_ib_mask] - exp_price, 0.0) - np.maximum(
                sb[ic_ib_mask] - exp_price, 0.0
            )
            cp = np.maximum(exp_price - sc[ic_ib_mask], 0.0) - np.maximum(
                exp_price - sd[ic_ib_mask], 0.0
            )
            payoff[ic_ib_mask] = pp + cp
        return payoff


def _gen_spreads_vec(
    strikes: np.ndarray,
    mids: np.ndarray,
    otm_pct: np.ndarray,
    spread_widths: List[int],
    *,
    is_put: bool,
    max_n: int,
    bids: np.ndarray,
    asks: np.ndarray,
) -> tuple[int, Dict[str, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
    """Vectorized vertical spread generation (PCS or CCS)."""
    valid_otm = (otm_pct >= 0) & (otm_pct <= MAX_OTM_PCT)
    short_s = strikes[valid_otm]
    short_m = mids[valid_otm]
    short_o = otm_pct[valid_otm]
    short_b = bids[valid_otm]
    short_a = asks[valid_otm]

    if len(short_s) == 0:
        return 0, {}, {}

    widths_arr = np.array(spread_widths, dtype=np.float64)

    if is_put:
        long_s_2d = short_s[:, None] - widths_arr[None, :]
    else:
        long_s_2d = short_s[:, None] + widths_arr[None, :]

    flat_long = long_s_2d.ravel()
    pos = np.searchsorted(strikes, flat_long)
    safe_pos = np.clip(pos, 0, max(len(strikes) - 1, 0))
    found = (pos < len(strikes)) & (strikes[safe_pos] == flat_long)
    long_m_flat = np.where(found, mids[safe_pos], np.nan)
    long_m_2d = long_m_flat.reshape(long_s_2d.shape)
    long_b_2d = np.where(found, bids[safe_pos], np.nan).reshape(long_s_2d.shape)
    long_a_2d = np.where(found, asks[safe_pos], np.nan).reshape(long_s_2d.shape)

    credits_2d = short_m[:, None] - long_m_2d
    valid_2d = np.isfinite(credits_2d) & (credits_2d > 0)

    ri, ci = np.where(valid_2d)
    n_valid = len(ri)
    if n_valid == 0:
        return 0, {}, {}

    if n_valid > max_n:
        ri, ci = ri[:max_n], ci[:max_n]
        n_valid = max_n

    flat_credits = credits_2d[ri, ci]
    flat_short_s = short_s[ri]
    flat_short_o = short_o[ri]
    flat_long_s = long_s_2d[ri, ci]
    flat_widths = widths_arr[ci]
    # Position bid = short_bid - long_ask, ask = short_ask - long_bid
    flat_pos_bid = short_b[ri] - long_a_2d[ri, ci]
    flat_pos_ask = short_a[ri] - long_b_2d[ri, ci]

    arrs: Dict[str, np.ndarray] = {
        "credit": flat_credits,
        "width": flat_widths,
        "param_a": flat_short_o,
        "strike_a": flat_short_s,
        "strike_b": flat_long_s,
        "bid": flat_pos_bid,
        "ask": flat_pos_ask,
    }

    by_width: Dict[int, Dict[str, np.ndarray]] = {}
    for w_idx, w in enumerate(spread_widths):
        mask = ci == w_idx
        if mask.any():
            by_width[int(w)] = {
                "short_strike": flat_short_s[mask],
                "short_otm": flat_short_o[mask],
                "credit": flat_credits[mask],
                "long_strike": flat_long_s[mask],
                "bid": flat_pos_bid[mask],
                "ask": flat_pos_ask[mask],
            }

    return n_valid, arrs, by_width


def _gen_ic_vec(
    pcs_bw: Dict[int, Dict[str, np.ndarray]],
    ccs_bw: Dict[int, Dict[str, np.ndarray]],
    max_n: int,
) -> tuple[int, Dict[str, np.ndarray]]:
    """Vectorized iron condor generation (PCS x CCS per width)."""
    parts: List[Dict[str, np.ndarray]] = []
    total = 0

    for w in SPREAD_WIDTHS:
        pw = pcs_bw.get(w)
        cw = ccs_bw.get(w)
        if pw is None or cw is None:
            continue

        n_p, n_c = len(pw["short_strike"]), len(cw["short_strike"])
        if n_p == 0 or n_c == 0:
            continue

        pi, ci = np.meshgrid(np.arange(n_p), np.arange(n_c), indexing="ij")
        pi, ci = pi.ravel(), ci.ravel()

        valid = pw["short_strike"][pi] < cw["short_strike"][ci]
        pi, ci = pi[valid], ci[valid]

        n_ic = len(pi)
        if n_ic == 0:
            continue

        space = max_n - total
        if n_ic > space:
            pi, ci = pi[:space], ci[:space]
            n_ic = space

        parts.append(
            {
                "credit": pw["credit"][pi] + cw["credit"][ci],
                "width": np.full(n_ic, w, dtype=np.float64),
                "param_a": pw["short_otm"][pi],
                "param_b": cw["short_otm"][ci],
                "strike_a": pw["short_strike"][pi],
                "strike_b": pw["long_strike"][pi],
                "strike_c": cw["short_strike"][ci],
                "strike_d": cw["long_strike"][ci],
                "bid": pw["bid"][pi] + cw["bid"][ci],
                "ask": pw["ask"][pi] + cw["ask"][ci],
            }
        )
        total += n_ic
        if total >= max_n:
            break

    if total == 0:
        return 0, {}

    merged: Dict[str, np.ndarray] = {}
    for key in parts[0]:
        merged[key] = np.concatenate([p[key] for p in parts])
    return total, merged


def _gen_ib_vec(
    strikes_all: np.ndarray,
    put_strikes: np.ndarray,
    put_mids: np.ndarray,
    call_strikes: np.ndarray,
    call_mids: np.ndarray,
    und_price: float,
    max_n: int,
    put_bids: np.ndarray,
    put_asks: np.ndarray,
    call_bids: np.ndarray,
    call_asks: np.ndarray,
) -> tuple[int, Dict[str, np.ndarray]]:
    """Vectorized iron butterfly generation."""
    centers = np.unique(strikes_all)
    center_off = (centers - und_price) / und_price
    valid_center = np.abs(center_off) <= MAX_OTM_PCT
    centers = centers[valid_center]
    center_off = center_off[valid_center]

    if len(centers) == 0 or len(put_strikes) == 0 or len(call_strikes) == 0:
        return 0, {}

    p_pos = np.searchsorted(put_strikes, centers)
    p_safe = np.clip(p_pos, 0, max(len(put_strikes) - 1, 0))
    p_found = (p_pos < len(put_strikes)) & (put_strikes[p_safe] == centers)

    c_pos = np.searchsorted(call_strikes, centers)
    c_safe = np.clip(c_pos, 0, max(len(call_strikes) - 1, 0))
    c_found = (c_pos < len(call_strikes)) & (call_strikes[c_safe] == centers)

    both = p_found & c_found
    centers = centers[both]
    center_off = center_off[both]
    sp_mids = put_mids[p_safe[both]]
    sc_mids = call_mids[c_safe[both]]
    sp_bids = put_bids[p_safe[both]]
    sp_asks = put_asks[p_safe[both]]
    sc_bids = call_bids[c_safe[both]]
    sc_asks = call_asks[c_safe[both]]

    if len(centers) == 0:
        return 0, {}

    widths_arr = np.array(SPREAD_WIDTHS, dtype=np.float64)
    lp_s_2d = centers[:, None] - widths_arr[None, :]
    lc_s_2d = centers[:, None] + widths_arr[None, :]

    lp_flat = lp_s_2d.ravel()
    lp_pos = np.searchsorted(put_strikes, lp_flat)
    lp_safe = np.clip(lp_pos, 0, max(len(put_strikes) - 1, 0))
    lp_found = (lp_pos < len(put_strikes)) & (put_strikes[lp_safe] == lp_flat)
    lp_mids = np.where(lp_found, put_mids[lp_safe], np.nan).reshape(lp_s_2d.shape)
    lp_bids = np.where(lp_found, put_bids[lp_safe], np.nan).reshape(lp_s_2d.shape)
    lp_asks = np.where(lp_found, put_asks[lp_safe], np.nan).reshape(lp_s_2d.shape)

    lc_flat = lc_s_2d.ravel()
    lc_pos = np.searchsorted(call_strikes, lc_flat)
    lc_safe = np.clip(lc_pos, 0, max(len(call_strikes) - 1, 0))
    lc_found = (lc_pos < len(call_strikes)) & (call_strikes[lc_safe] == lc_flat)
    lc_mids = np.where(lc_found, call_mids[lc_safe], np.nan).reshape(lc_s_2d.shape)
    lc_bids = np.where(lc_found, call_bids[lc_safe], np.nan).reshape(lc_s_2d.shape)
    lc_asks = np.where(lc_found, call_asks[lc_safe], np.nan).reshape(lc_s_2d.shape)

    put_cr = sp_mids[:, None] - lp_mids
    call_cr = sc_mids[:, None] - lc_mids
    valid = np.isfinite(put_cr) & np.isfinite(call_cr) & (put_cr > 0) & (call_cr > 0)

    ri, ci = np.where(valid)
    n_valid = len(ri)
    if n_valid == 0:
        return 0, {}
    if n_valid > max_n:
        ri, ci = ri[:max_n], ci[:max_n]
        n_valid = max_n

    # Position bid = short_bid - long_ask per leg, summed
    put_pos_bid = sp_bids[ri] - lp_asks[ri, ci]
    put_pos_ask = sp_asks[ri] - lp_bids[ri, ci]
    call_pos_bid = sc_bids[ri] - lc_asks[ri, ci]
    call_pos_ask = sc_asks[ri] - lc_bids[ri, ci]

    return n_valid, {
        "credit": put_cr[ri, ci] + call_cr[ri, ci],
        "width": widths_arr[ci],
        "param_a": center_off[ri],
        "strike_a": centers[ri],
        "strike_b": lp_s_2d[ri, ci],
        "strike_c": centers[ri].copy(),
        "strike_d": lc_s_2d[ri, ci],
        "bid": put_pos_bid + call_pos_bid,
        "ask": put_pos_ask + call_pos_ask,
    }


def generate_candidates_batch(
    und_price: float,
    chain: pd.DataFrame,
    max_cand: int = MAX_CANDIDATES_PER_MINUTE,
) -> CandidateBatch | None:
    """Vectorized candidate generation returning columnar CandidateBatch."""
    if max_cand <= 0 or chain.empty or not np.isfinite(und_price) or und_price <= 0:
        return None

    strikes_all = chain["strike"].to_numpy(dtype=np.float64, copy=False)
    mids_all = chain["mid"].to_numpy(dtype=np.float64, copy=False)
    is_put_all = chain["is_put"].to_numpy(dtype=bool, copy=False)
    has_ba = "bid" in chain.columns and "ask" in chain.columns
    bids_all = chain["bid"].to_numpy(dtype=np.float64, copy=False) if has_ba else None
    asks_all = chain["ask"].to_numpy(dtype=np.float64, copy=False) if has_ba else None

    put_strikes, put_mids, put_otm, put_bids, put_asks = _prepare_option_quotes(
        strikes_all,
        mids_all,
        is_put_all,
        und_price,
        is_put=True,
        bids_all=bids_all,
        asks_all=asks_all,
    )
    call_strikes, call_mids, call_otm, call_bids, call_asks = _prepare_option_quotes(
        strikes_all,
        mids_all,
        is_put_all,
        und_price,
        is_put=False,
        bids_all=bids_all,
        asks_all=asks_all,
    )

    remaining = max_cand
    sections: List[tuple[int, int, Dict[str, np.ndarray]]] = []
    pcs_bw: Dict[int, Dict[str, np.ndarray]] = {}
    ccs_bw: Dict[int, Dict[str, np.ndarray]] = {}

    n_pcs, pcs_arrs, pcs_bw = _gen_spreads_vec(
        put_strikes,
        put_mids,
        put_otm,
        SPREAD_WIDTHS,
        is_put=True,
        max_n=remaining,
        bids=put_bids,
        asks=put_asks,
    )
    if n_pcs > 0:
        sections.append((_TYPE_PCS, n_pcs, pcs_arrs))
        remaining -= n_pcs

    if remaining > 0:
        n_ccs, ccs_arrs, ccs_bw = _gen_spreads_vec(
            call_strikes,
            call_mids,
            call_otm,
            SPREAD_WIDTHS,
            is_put=False,
            max_n=remaining,
            bids=call_bids,
            asks=call_asks,
        )
        if n_ccs > 0:
            sections.append((_TYPE_CCS, n_ccs, ccs_arrs))
            remaining -= n_ccs

    if remaining > 0 and pcs_bw and ccs_bw:
        n_ic, ic_arrs = _gen_ic_vec(pcs_bw, ccs_bw, remaining)
        if n_ic > 0:
            sections.append((_TYPE_IC, n_ic, ic_arrs))
            remaining -= n_ic

    if remaining > 0:
        n_ib, ib_arrs = _gen_ib_vec(
            strikes_all,
            put_strikes,
            put_mids,
            call_strikes,
            call_mids,
            und_price,
            remaining,
            put_bids=put_bids,
            put_asks=put_asks,
            call_bids=call_bids,
            call_asks=call_asks,
        )
        if n_ib > 0:
            sections.append((_TYPE_IB, n_ib, ib_arrs))

    if not sections:
        return None

    total = sum(n for _, n, _ in sections)
    # Assemble in float64 first for precision parity with dict-based code
    credit_f64 = np.empty(total, dtype=np.float64)
    width_f64 = np.empty(total, dtype=np.float64)
    param_a_f64 = np.zeros(total, dtype=np.float64)
    param_b_f64 = np.zeros(total, dtype=np.float64)
    type_id = np.empty(total, dtype=np.int8)
    strike_a = np.zeros(total, dtype=np.float64)
    strike_b = np.zeros(total, dtype=np.float64)
    strike_c = np.zeros(total, dtype=np.float64)
    strike_d = np.zeros(total, dtype=np.float64)
    bid_f64 = np.zeros(total, dtype=np.float64)
    ask_f64 = np.zeros(total, dtype=np.float64)

    type_ranges: Dict[str, tuple[int, int]] = {}
    pos = 0
    for tid, n_t, arrs in sections:
        s, e = pos, pos + n_t
        type_ranges[_TYPE_NAMES[tid]] = (s, e)
        type_id[s:e] = tid
        credit_f64[s:e] = arrs["credit"]
        width_f64[s:e] = arrs["width"]
        param_a_f64[s:e] = arrs["param_a"]
        if "param_b" in arrs:
            param_b_f64[s:e] = arrs["param_b"]
        strike_a[s:e] = arrs["strike_a"]
        strike_b[s:e] = arrs["strike_b"]
        if "strike_c" in arrs:
            strike_c[s:e] = arrs["strike_c"]
        if "strike_d" in arrs:
            strike_d[s:e] = arrs["strike_d"]
        bid_f64[s:e] = arrs["bid"]
        ask_f64[s:e] = arrs["ask"]
        pos = e

    # Convert to float32 after float64 assembly (matches old code precision)
    credit = credit_f64.astype(np.float32)
    width = width_f64.astype(np.float32)
    param_a = param_a_f64.astype(np.float32)
    param_b = param_b_f64.astype(np.float32)
    risk = (width_f64 - credit_f64).astype(np.float32)

    return CandidateBatch(
        n=total,
        credit=credit,
        risk=risk,
        width=width,
        param_a=param_a,
        param_b=param_b,
        type_id=type_id,
        type_ranges=type_ranges,
        strike_a=strike_a,
        strike_b=strike_b,
        strike_c=strike_c,
        strike_d=strike_d,
        credit_f64=credit_f64,
        width_f64=width_f64,
        bid_f64=bid_f64,
        ask_f64=ask_f64,
    )


def _oracle_gpu_submit(
    batch: CandidateBatch,
    past_unds_t: torch.Tensor,
    past_exps_t: torch.Tensor,
) -> torch.Tensor:
    """Submit GPU oracle work (async). Returns GPU tensor [2, n]."""
    n = batch.n
    n_hist = past_unds_t.shape[0]

    params = torch.as_tensor(batch.gpu_params, device=DEVICE)
    credit_t = params[:, 0]
    risk_t = params[:, 1]
    pa = params[:, 2]
    pb = params[:, 3]
    wd = params[:, 4]

    exp_r = past_exps_t.unsqueeze(0)
    und_r = past_unds_t.unsqueeze(0)

    payoff = torch.empty(n, n_hist, device=DEVICE, dtype=torch.float32)

    for type_name, (s, e) in batch.type_ranges.items():
        if s == e:
            continue
        if type_name == "pcs":
            sh = und_r * (1.0 - pa[s:e].unsqueeze(1))
            lh = sh - wd[s:e].unsqueeze(1)
            payoff[s:e] = torch.clamp(sh - exp_r, min=0.0) - torch.clamp(
                lh - exp_r, min=0.0
            )
        elif type_name == "ccs":
            sh = und_r * (1.0 + pa[s:e].unsqueeze(1))
            lh = sh + wd[s:e].unsqueeze(1)
            payoff[s:e] = torch.clamp(exp_r - sh, min=0.0) - torch.clamp(
                exp_r - lh, min=0.0
            )
        elif type_name == "ic":
            sph = und_r * (1.0 - pa[s:e].unsqueeze(1))
            lph = sph - wd[s:e].unsqueeze(1)
            sch = und_r * (1.0 + pb[s:e].unsqueeze(1))
            lch = sch + wd[s:e].unsqueeze(1)
            payoff[s:e] = (
                torch.clamp(sph - exp_r, min=0.0)
                - torch.clamp(lph - exp_r, min=0.0)
                + torch.clamp(exp_r - sch, min=0.0)
                - torch.clamp(exp_r - lch, min=0.0)
            )
        elif type_name == "ib":
            ch = und_r * (1.0 + pa[s:e].unsqueeze(1))
            lph = ch - wd[s:e].unsqueeze(1)
            lch = ch + wd[s:e].unsqueeze(1)
            payoff[s:e] = (
                torch.clamp(ch - exp_r, min=0.0)
                - torch.clamp(lph - exp_r, min=0.0)
                + torch.clamp(exp_r - ch, min=0.0)
                - torch.clamp(exp_r - lch, min=0.0)
            )
        else:
            raise ValueError(f"Unknown type: {type_name}")

    pnl = credit_t.unsqueeze(1) - payoff
    wr = (pnl > 0).float().mean(dim=1)
    ar = (pnl / risk_t.unsqueeze(1)).mean(dim=1)

    # Return GPU tensor — no .cpu() yet (stays async)
    return torch.stack([wr, ar])


def _oracle_gpu_gather(gpu_result: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Sync GPU→CPU transfer and return numpy arrays."""
    result = gpu_result.cpu().numpy()
    return result[0].copy(), result[1].copy()


def oracle_batch_stats(
    current_dt: pd.Timestamp,
    candidates: List[Dict],
    static_data: BacktestStaticData,
    lookback: int = LOOKBACK_DAYS,
) -> List[Dict]:
    if not candidates:
        return []

    current_ord = current_dt.date().toordinal()
    current_sec = current_dt.hour * 3600 + current_dt.minute * 60
    current_day_index = static_data.day_ord_to_index.get(current_ord)
    minute_index = static_data.minute_sec_to_index.get(current_sec)
    if current_day_index is None or minute_index is None:
        return [{} for _ in candidates]

    start_index = max(0, current_day_index - lookback)
    if current_day_index - start_index < MIN_HISTORY_OBS:
        return [{} for _ in candidates]

    past_unds_t = static_data.hist_und_tensor[
        start_index:current_day_index, minute_index
    ]
    past_exps_t = static_data.hist_exp_tensor[start_index:current_day_index]
    valid_mask = torch.isfinite(past_unds_t) & torch.isfinite(past_exps_t)
    n_hist = int(valid_mask.sum().item())
    if n_hist < MIN_HISTORY_OBS:
        return [{} for _ in candidates]

    past_unds_t = past_unds_t[valid_mask]
    past_exps_t = past_exps_t[valid_mask]

    groups: Dict[str, List[tuple[int, Dict]]] = {
        "pcs": [],
        "ccs": [],
        "ic": [],
        "ib": [],
    }
    for i, candidate in enumerate(candidates):
        groups[candidate["type"]].append((i, candidate))

    results: List[Dict] = [{} for _ in candidates]

    with torch.no_grad():
        for gtype, group in groups.items():
            if not group:
                continue

            idxs = [x[0] for x in group]
            cands = [x[1] for x in group]
            n_candidates = len(cands)

            credit_t = torch.tensor(
                [c["credit"] for c in cands], device=DEVICE, dtype=torch.float32
            )
            risk_t = torch.tensor(
                [c["risk"] for c in cands], device=DEVICE, dtype=torch.float32
            )

            if gtype == "pcs":
                short_otm_t = torch.tensor(
                    [c["short_otm"] for c in cands], device=DEVICE, dtype=torch.float32
                )
                width_t = torch.tensor(
                    [c["width"] for c in cands], device=DEVICE, dtype=torch.float32
                )

                short_hist = past_unds_t.unsqueeze(0) * (1.0 - short_otm_t.unsqueeze(1))
                long_hist = short_hist - width_t.unsqueeze(1)
                payoff = torch.clamp(
                    short_hist - past_exps_t.unsqueeze(0), min=0.0
                ) - torch.clamp(long_hist - past_exps_t.unsqueeze(0), min=0.0)

            elif gtype == "ccs":
                short_otm_t = torch.tensor(
                    [c["short_otm"] for c in cands], device=DEVICE, dtype=torch.float32
                )
                width_t = torch.tensor(
                    [c["width"] for c in cands], device=DEVICE, dtype=torch.float32
                )

                short_hist = past_unds_t.unsqueeze(0) * (1.0 + short_otm_t.unsqueeze(1))
                long_hist = short_hist + width_t.unsqueeze(1)
                payoff = torch.clamp(
                    past_exps_t.unsqueeze(0) - short_hist, min=0.0
                ) - torch.clamp(past_exps_t.unsqueeze(0) - long_hist, min=0.0)

            elif gtype == "ic":
                put_short_otm_t = torch.tensor(
                    [c["put_short_otm"] for c in cands],
                    device=DEVICE,
                    dtype=torch.float32,
                )
                call_short_otm_t = torch.tensor(
                    [c["call_short_otm"] for c in cands],
                    device=DEVICE,
                    dtype=torch.float32,
                )
                width_t = torch.tensor(
                    [c["width"] for c in cands], device=DEVICE, dtype=torch.float32
                )

                short_put_hist = past_unds_t.unsqueeze(0) * (
                    1.0 - put_short_otm_t.unsqueeze(1)
                )
                long_put_hist = short_put_hist - width_t.unsqueeze(1)

                short_call_hist = past_unds_t.unsqueeze(0) * (
                    1.0 + call_short_otm_t.unsqueeze(1)
                )
                long_call_hist = short_call_hist + width_t.unsqueeze(1)

                put_pay = torch.clamp(
                    short_put_hist - past_exps_t.unsqueeze(0), min=0.0
                ) - torch.clamp(long_put_hist - past_exps_t.unsqueeze(0), min=0.0)
                call_pay = torch.clamp(
                    past_exps_t.unsqueeze(0) - short_call_hist, min=0.0
                ) - torch.clamp(past_exps_t.unsqueeze(0) - long_call_hist, min=0.0)
                payoff = put_pay + call_pay

            elif gtype == "ib":
                center_offset_t = torch.tensor(
                    [c["center_offset_pct"] for c in cands],
                    device=DEVICE,
                    dtype=torch.float32,
                )
                width_t = torch.tensor(
                    [c["width"] for c in cands], device=DEVICE, dtype=torch.float32
                )

                center_hist = past_unds_t.unsqueeze(0) * (
                    1.0 + center_offset_t.unsqueeze(1)
                )
                long_put_hist = center_hist - width_t.unsqueeze(1)
                long_call_hist = center_hist + width_t.unsqueeze(1)

                put_pay = torch.clamp(
                    center_hist - past_exps_t.unsqueeze(0), min=0.0
                ) - torch.clamp(long_put_hist - past_exps_t.unsqueeze(0), min=0.0)
                call_pay = torch.clamp(
                    past_exps_t.unsqueeze(0) - center_hist, min=0.0
                ) - torch.clamp(past_exps_t.unsqueeze(0) - long_call_hist, min=0.0)
                payoff = put_pay + call_pay

            else:
                raise ValueError(f"Unknown group type: {gtype}")

            pnl = credit_t.unsqueeze(1) - payoff
            win_rate = (pnl > 0).float().mean(dim=1).cpu().numpy()
            avg_ror = (pnl / risk_t.unsqueeze(1)).mean(dim=1).cpu().numpy()

            for i_local in range(n_candidates):
                orig_idx = idxs[i_local]
                results[orig_idx] = {
                    **cands[i_local],
                    "win_rate": float(win_rate[i_local]),
                    "avg_ror": float(avg_ror[i_local]),
                    "n_hist": n_hist,
                }

    return results


def summarize_results(df_cal: pd.DataFrame) -> pd.DataFrame:
    """Build the calibration summary used by both performance runs and analysis."""
    working = df_cal.assign(
        ror_sq=df_cal["ror"] ** 2,
        win_ror=df_cal["ror"].where(df_cal["ror"] > 0),
        loss_ror=df_cal["ror"].where(df_cal["ror"] < 0),
        downside_ror=df_cal["ror"].where(df_cal["ror"] < 0),
        downside_sq=(df_cal["ror"].where(df_cal["ror"] < 0)) ** 2,
    )

    grouped = working.groupby(["minute", "min_wr_thresh", "min_ror_thresh"], sort=False)
    summary = grouped.agg(
        n_trades=("ror", "count"),
        predicted_win_rate=("predicted_wr", "mean"),
        win_rate=("actual_win", "mean"),
        avg_ror=("ror", "mean"),
        predicted_ror=("predicted_ror", "mean"),
        avg_win=("win_ror", "mean"),
        avg_loss=("loss_ror", "mean"),
        avg_pnl_pts=("pnl_pts", "mean"),
        avg_credit_pts=("credit_pts", "mean"),
        avg_risk_pts=("risk_pts", "mean"),
        avg_pnl_dollars=("pnl_dollars", "mean"),
        avg_credit_dollars=("credit_dollars", "mean"),
        avg_risk_dollars=("risk_dollars", "mean"),
        avg_hist_obs=("n_hist", "mean"),
        ror_sum=("ror", "sum"),
        ror_sq_sum=("ror_sq", "sum"),
        downside_count=("downside_ror", "count"),
        downside_sum=("downside_ror", "sum"),
        downside_sq_sum=("downside_sq", "sum"),
    )

    n_trades = summary["n_trades"].to_numpy(dtype=np.float64, copy=False)
    avg_ror = summary["avg_ror"].to_numpy(dtype=np.float64, copy=False)
    ror_sum = summary["ror_sum"].to_numpy(dtype=np.float64, copy=False)
    ror_sq_sum = summary["ror_sq_sum"].to_numpy(dtype=np.float64, copy=False)
    downside_count = summary["downside_count"].to_numpy(dtype=np.float64, copy=False)
    downside_sum = summary["downside_sum"].to_numpy(dtype=np.float64, copy=False)
    downside_sq_sum = summary["downside_sq_sum"].to_numpy(dtype=np.float64, copy=False)

    sharpe = np.zeros_like(avg_ror)
    sharpe_mask = n_trades >= 2
    sharpe_var = np.zeros_like(avg_ror)
    sharpe_var[sharpe_mask] = (
        ror_sq_sum[sharpe_mask] - (ror_sum[sharpe_mask] ** 2) / n_trades[sharpe_mask]
    ) / (n_trades[sharpe_mask] - 1.0)
    sharpe_std = np.sqrt(np.maximum(sharpe_var, 0.0))
    valid_sharpe = np.isfinite(sharpe_std) & (sharpe_std > 0)
    sharpe[valid_sharpe] = (
        avg_ror[valid_sharpe] / sharpe_std[valid_sharpe] * np.sqrt(252)
    )

    sortino = np.zeros_like(avg_ror)
    sortino_mask = downside_count >= 2
    downside_var = np.zeros_like(avg_ror)
    downside_var[sortino_mask] = (
        downside_sq_sum[sortino_mask]
        - (downside_sum[sortino_mask] ** 2) / downside_count[sortino_mask]
    ) / (downside_count[sortino_mask] - 1.0)
    downside_std = np.sqrt(np.maximum(downside_var, 0.0))
    valid_sortino = np.isfinite(downside_std) & (downside_std > 0)
    sortino[valid_sortino] = (
        avg_ror[valid_sortino] / downside_std[valid_sortino] * np.sqrt(252)
    )

    summary["avg_win"] = summary["avg_win"].fillna(0.0)
    summary["avg_loss"] = summary["avg_loss"].fillna(0.0)
    summary["sharpe"] = sharpe
    summary["sortino"] = sortino

    return summary[
        [
            "n_trades",
            "predicted_win_rate",
            "win_rate",
            "avg_ror",
            "predicted_ror",
            "avg_win",
            "avg_loss",
            "avg_pnl_pts",
            "avg_credit_pts",
            "avg_risk_pts",
            "avg_pnl_dollars",
            "avg_credit_dollars",
            "avg_risk_dollars",
            "avg_hist_obs",
            "sharpe",
            "sortino",
        ]
    ].reset_index()


def run_analysis_and_visualization(
    df_cal: pd.DataFrame, res_summary: pd.DataFrame
) -> None:
    """Keep the expensive post-processing separate from the walk-forward core."""
    print("\n=== BEST COMBINATIONS (Sharpe) ===")
    print(
        res_summary.nlargest(5, "sharpe")[
            [
                "minute",
                "min_wr_thresh",
                "min_ror_thresh",
                "sharpe",
                "sortino",
                "predicted_win_rate",
                "predicted_ror",
                "win_rate",
                "avg_ror",
                "avg_win",
                "avg_loss",
                "avg_pnl_pts",
                "avg_credit_pts",
                "avg_risk_pts",
                "avg_pnl_dollars",
                "avg_credit_dollars",
                "avg_risk_dollars",
                "avg_hist_obs",
                "n_trades",
            ]
        ]
    )

    for ror_thresh in MIN_ROR_THRESHOLDS:
        subset = res_summary[res_summary["min_ror_thresh"] == ror_thresh]
        if subset.empty:
            continue
        pivot = subset.pivot(index="min_wr_thresh", columns="minute", values="sharpe")
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(
            pivot, annot=False, fmt=".2f", cmap="RdYlGn", center=0.0, robust=True
        )
        ax.set_yticklabels([f"{y:.0%}" for y in pivot.index])
        plt.title(
            f"Sharpe Ratio Heatmap \u2013 SPXW 0DTE Oracle"
            f" (min ROR \u2265 {ror_thresh:.0%})"
        )
        plt.tight_layout()
        plt.show()

    n_bins = 100

    df_wr = df_cal.copy()
    df_wr["wr_bin"] = pd.cut(
        df_wr["predicted_wr"], bins=n_bins, labels=False, include_lowest=True
    )
    calibration_wr = (
        df_wr.groupby("wr_bin")
        .agg(
            pred_mean=("predicted_wr", "mean"),
            actual=("actual_win", "mean"),
            count=("actual_win", "count"),
        )
        .dropna()
        .reset_index()
    )

    plt.figure(figsize=(12, 12))
    plt.scatter(
        calibration_wr["pred_mean"],
        calibration_wr["actual"],
        s=calibration_wr["count"] / 2,
        alpha=0.7,
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Oracle Win-Rate Calibration")
    plt.xlabel("Predicted win rate")
    plt.ylabel("Actual win rate")
    plt.tight_layout()
    plt.show()

    print(f"Overall predicted WR: {df_cal['predicted_wr'].mean():.1%}")
    print(f"Actual win rate     : {df_cal['actual_win'].mean():.1%}")
    print(
        f"Brier Score         : "
        f"{((df_cal['predicted_wr'] - df_cal['actual_win']) ** 2).mean():.4f}"
    )

    df_ror = df_cal.copy()
    df_ror["ror_bin"] = pd.cut(
        df_ror["predicted_ror"], bins=n_bins, labels=False, include_lowest=True
    )
    calibration_ror = (
        df_ror.groupby("ror_bin")
        .agg(
            pred_mean=("predicted_ror", "mean"),
            actual=("ror", "mean"),
            count=("ror", "count"),
        )
        .dropna()
        .reset_index()
    )

    plt.figure(figsize=(12, 12))
    plt.scatter(
        calibration_ror["pred_mean"],
        calibration_ror["actual"],
        s=calibration_ror["count"] / 2,
        alpha=0.7,
    )

    if not calibration_ror.empty:
        line_min = float(
            min(calibration_ror["pred_mean"].min(), calibration_ror["actual"].min())
        )
        line_max = float(
            max(calibration_ror["pred_mean"].max(), calibration_ror["actual"].max())
        )
        if np.isfinite(line_min) and np.isfinite(line_max) and line_max > line_min:
            plt.plot([line_min, line_max], [line_min, line_max], "k--")

    plt.title("Oracle ROR Calibration")
    plt.xlabel("Predicted ROR")
    plt.ylabel("Actual ROR")
    plt.tight_layout()
    plt.show()

    print(f"Overall predicted ROR: {df_cal['predicted_ror'].mean():.4f}")
    print(f"Overall actual ROR   : {df_cal['ror'].mean():.4f}")
    print(
        f"ROR MSE              : "
        f"{((df_cal['predicted_ror'] - df_cal['ror']) ** 2).mean():.6f}"
    )


def process_day_fast(
    tdate: date,
    day_chain_by_dt: Dict[pd.Timestamp, pd.DataFrame],
    exp_p: float,
    static_data: BacktestStaticData,
) -> List[Dict]:
    """Process all minutes using pre-computed day-level history (per-minute GPU)."""
    current_ord = tdate.toordinal()
    current_day_index = static_data.day_ord_to_index.get(current_ord)
    if current_day_index is None:
        return []

    start_index = max(0, current_day_index - LOOKBACK_DAYS)
    if current_day_index - start_index < MIN_HISTORY_OBS:
        return []

    # Pre-compute once per day
    past_exps_full = static_data.hist_exp_tensor[start_index:current_day_index]
    past_unds_full = static_data.hist_und_tensor[start_index:current_day_index, :]
    exp_valid = torch.isfinite(past_exps_full)

    # Pre-compute per-minute filtered history
    minute_hist_cache: Dict[time, tuple[torch.Tensor, torch.Tensor, int]] = {}
    for minute in TEST_MINUTES:
        m_sec = minute.hour * 3600 + minute.minute * 60
        m_idx = static_data.minute_sec_to_index.get(m_sec)
        if m_idx is None:
            continue
        past_unds_m = past_unds_full[:, m_idx]
        valid_mask = torch.isfinite(past_unds_m) & exp_valid
        n_hist = int(valid_mask.sum().item())
        if n_hist < MIN_HISTORY_OBS:
            continue
        minute_hist_cache[minute] = (
            past_unds_m[valid_mask],
            past_exps_full[valid_mask],
            n_hist,
        )

    trades: List[Dict] = []

    # Collect valid minutes with their cached data
    valid_minutes: List[
        tuple[time, pd.DataFrame, float, torch.Tensor, torch.Tensor, int]
    ] = []
    for minute in TEST_MINUTES:
        cached = minute_hist_cache.get(minute)
        if cached is None:
            continue
        entry_dt = get_timestamp(tdate, minute)
        chain = day_chain_by_dt.get(entry_dt)
        if chain is None or len(chain) < 10:
            continue
        und_price = float(chain["und_price"].iloc[0])
        if not np.isfinite(und_price) or und_price <= 0:
            continue
        past_unds_t, past_exps_t, n_hist = cached
        valid_minutes.append(
            (minute, chain, und_price, past_unds_t, past_exps_t, n_hist)
        )

    if not valid_minutes:
        return trades

    with torch.no_grad():
        # Pipeline: overlap GPU compute of current minute with
        # CandGen of next minute
        prev_batch: CandidateBatch | None = None
        prev_gpu_result: torch.Tensor | None = None
        prev_minute: time | None = None
        prev_n_hist: int = 0

        for vi in range(len(valid_minutes)):
            minute, chain, und_price, past_unds_t, past_exps_t, n_hist = valid_minutes[
                vi
            ]

            # Generate candidates (CPU — overlaps with prev GPU compute)
            batch = generate_candidates_batch(und_price, chain)

            # Gather PREVIOUS minute's GPU results + trade selection
            if (
                prev_gpu_result is not None
                and prev_batch is not None
                and prev_minute is not None
            ):
                wr_arr, ar_arr = _oracle_gpu_gather(prev_gpu_result)
                _select_trades(
                    trades,
                    prev_batch,
                    wr_arr,
                    ar_arr,
                    prev_minute,
                    prev_n_hist,
                    tdate,
                    exp_p,
                )
                prev_gpu_result = None

            # Submit THIS minute to GPU (async — returns immediately)
            if batch is not None:
                prev_gpu_result = _oracle_gpu_submit(batch, past_unds_t, past_exps_t)
                prev_batch = batch
                prev_minute = minute
                prev_n_hist = n_hist
            else:
                prev_batch = None
                prev_gpu_result = None

        # Process last minute
        if (
            prev_gpu_result is not None
            and prev_batch is not None
            and prev_minute is not None
        ):
            wr_arr, ar_arr = _oracle_gpu_gather(prev_gpu_result)
            _select_trades(
                trades,
                prev_batch,
                wr_arr,
                ar_arr,
                prev_minute,
                prev_n_hist,
                tdate,
                exp_p,
            )

    return trades


def _select_trades(
    trades: List[Dict],
    batch: CandidateBatch,
    wr_arr: np.ndarray,
    ar_arr: np.ndarray,
    minute: time,
    n_hist: int,
    tdate: date,
    exp_p: float,
) -> None:
    """Vectorized trade selection for one minute's results."""
    valid_mask = (batch.risk > 0) & np.isfinite(wr_arr) & np.isfinite(ar_arr)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return

    sorted_order = valid_idx[np.argsort(-ar_arr[valid_idx], kind="mergesort")]
    sorted_wr = wr_arr[sorted_order]

    passes = sorted_wr[:, None] >= MIN_WR_THRESHOLDS[None, :]
    has_any = passes.any(axis=0)
    if not has_any.any():
        return
    first_pos = passes.argmax(axis=0)

    active = np.where(has_any)[0]
    winner_pos = first_pos[active]
    winner_idx = sorted_order[winner_pos]

    payoff_arr = batch.payoffs_at(winner_idx, exp_p)
    if SLIPPAGE_PCT_OF_BASPREAD > 0:
        ba_spread = batch.ask_f64[winner_idx] - batch.bid_f64[winner_idx]
        credit_arr = batch.credit_f64[winner_idx] - ba_spread * SLIPPAGE_PCT_OF_BASPREAD
    else:
        credit_arr = batch.credit_f64[winner_idx]
    risk_arr = (batch.width_f64 - batch.credit_f64)[winner_idx]
    width_arr = batch.width_f64[winner_idx]
    pnl_arr = credit_arr - payoff_arr
    ror_arr = np.where(risk_arr > 0, pnl_arr / risk_arr, 0.0)
    type_ids = batch.type_id[winner_idx]

    minute_str = minute.strftime("%H:%M")
    for j in range(len(active)):
        ti = active[j]
        bi = int(winner_idx[j])
        pnl_pts = float(pnl_arr[j])
        credit_pts = float(credit_arr[j])
        risk_pts = float(risk_arr[j])
        pred_ror = float(ar_arr[bi])
        pred_wr = float(wr_arr[bi])
        actual_ror = float(ror_arr[j])
        stype = _TYPE_NAMES[int(type_ids[j])]
        width = float(width_arr[j])
        wr_thresh = float(MIN_WR_THRESHOLDS[ti])

        for min_ror in MIN_ROR_THRESHOLDS:
            if pred_ror >= min_ror:
                trades.append(
                    {
                        "date": tdate,
                        "minute": minute_str,
                        "min_wr_thresh": wr_thresh,
                        "min_ror_thresh": float(min_ror),
                        "predicted_wr": pred_wr,
                        "actual_win": int(pnl_pts > 0),
                        "ror": actual_ror,
                        "strategy_type": stype,
                        "width": width,
                        "predicted_ror": pred_ror,
                        "n_hist": int(n_hist),
                        "pnl_pts": pnl_pts,
                        "credit_pts": credit_pts,
                        "risk_pts": risk_pts,
                        "pnl_dollars": pnl_pts * CONTRACT_MULT,
                        "credit_dollars": credit_pts * CONTRACT_MULT,
                        "risk_dollars": risk_pts * CONTRACT_MULT,
                    }
                )
            else:
                trades.append(
                    {
                        "date": tdate,
                        "minute": minute_str,
                        "min_wr_thresh": wr_thresh,
                        "min_ror_thresh": float(min_ror),
                        "predicted_wr": pred_wr,
                        "actual_win": 0,
                        "ror": 0.0,
                        "strategy_type": "nt",
                        "width": 0.0,
                        "predicted_ror": pred_ror,
                        "n_hist": int(n_hist),
                        "pnl_pts": 0.0,
                        "credit_pts": 0.0,
                        "risk_pts": 0.0,
                        "pnl_dollars": 0.0,
                        "credit_dollars": 0.0,
                        "risk_dollars": 0.0,
                    }
                )


def main(run_analysis: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Using device: {DEVICE}")
    print("Connecting to Oracle...")
    engine = create_engine(f"oracle+oracledb://{user}:{password}@{dsn}")
    print("✅ Connected to Oracle successfully.")

    try:
        static_data = load_static_backtest_data(engine)

        master_trades: List[Dict] = []
        print("Starting walk-forward (day-fast oracle)...")

        import time as _time
        from concurrent.futures import ThreadPoolExecutor

        _wf_start = _time.perf_counter()
        test_days = static_data.test_days
        with ThreadPoolExecutor(max_workers=1) as sql_pool:
            # Kick off the first SQL load
            future = sql_pool.submit(load_live_spx_chain_day, engine, test_days[0])
            for i, tdate in enumerate(tqdm(test_days, desc="Walk-forward days")):
                day_chain_by_dt = future.result()
                # Prefetch next day while processing this one
                if i + 1 < len(test_days):
                    future = sql_pool.submit(
                        load_live_spx_chain_day, engine, test_days[i + 1]
                    )

                exp_p = static_data.exp_price.get(tdate)
                if pd.isna(exp_p):
                    continue

                trades = process_day_fast(
                    tdate, day_chain_by_dt, float(exp_p), static_data
                )
                master_trades.extend(trades)
        _wf_elapsed = _time.perf_counter() - _wf_start
        print(f"Walk-forward elapsed: {_wf_elapsed:.1f}s")

        df_cal = pd.DataFrame(master_trades)
        if df_cal.empty:
            print("No trades generated.")
            return pd.DataFrame(), pd.DataFrame()

        res_summary = summarize_results(df_cal)
        if run_analysis:
            run_analysis_and_visualization(df_cal, res_summary)

        return df_cal, res_summary
    finally:
        engine.dispose()
        print("✅ Script complete. Connection closed.")


if __name__ == "__main__":
    df_cal, res_summary = main()
