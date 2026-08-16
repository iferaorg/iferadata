"""Load strategy backtests, optimize their allocation, and report the result."""

# Public CSV facades intentionally mirror the tensor optimizer's parameters.
# pylint: disable=duplicate-code

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import torch
from holidays.financial.ny_stock_exchange import NYSE
from tqdm.auto import tqdm  # pylint: disable=unused-import

from ifera.settings import settings

from . import capital_allocation

__all__ = [
    "PortfolioAllocationResult",
    "PortfolioDiversification",
    "PortfolioStatistics",
    "WalkForwardFold",
    "WalkForwardPortfolioAllocationResult",
    "find_optimal_capital_allocation_from_csv",
    "walk_forward_capital_allocation_from_csv",
]

_ANNUALIZATION_PERIODS = 252
_CORRELATION_ZERO_TOLERANCE = 1e-12
_CSV_DATE_FORMATS = (
    "%b %-d, %Y %-I:%M%p",
    "%b %-d, %Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)
_RESULT_COLUMN_NAMES = frozenset(
    ("date", "active", "combined_return", "log_equity", "equity", "drawdown")
)


@dataclass(frozen=True)
class PortfolioStatistics:
    """Financial statistics for an optimized combined daily return series."""

    start_date: date
    end_date: date
    market_days: int
    active_days: int
    winning_days: int
    losing_days: int
    ending_equity: float
    total_return: float
    cagr: float
    cmgr: float
    win_rate: float
    average_drawdown: float
    bootstrap_average_drawdown: float | None
    max_drawdown: float
    max_drawdown_duration: int
    average_daily_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    exposure: float
    best_day: float
    worst_day: float


@dataclass(frozen=True)
class PortfolioDiversification:
    """Pairwise strategy correlations and allocation-concentration statistics."""

    correlation_matrix: torch.Tensor
    strategy_count: int
    allocated_strategy_count: int
    total_pair_count: int
    valid_pair_count: int
    highest_correlation: float | None
    highest_correlation_pair: tuple[str, str] | None
    lowest_negative_correlation: float | None
    lowest_negative_correlation_pair: tuple[str, str] | None
    mean_absolute_correlation: float | None
    mean_correlation: float | None
    allocation_concentration: float | None
    effective_strategy_count: float


@dataclass(frozen=True)
class PortfolioAllocationResult:
    """Allocation, synchronized daily series, and statistics returned by the wrapper."""

    strategy_names: tuple[str, ...]
    allocation: torch.Tensor
    daily_results: pl.DataFrame
    statistics: PortfolioStatistics
    diversification: PortfolioDiversification


@dataclass(frozen=True)
class WalkForwardFold:
    """One training window and its calculated and applied allocations."""

    training_start_date: date
    training_end_date: date
    simulation_start_date: date
    simulation_end_date: date
    calculated_allocation: torch.Tensor
    applied_allocation: torch.Tensor

    @property
    def allocation(self) -> torch.Tensor:
        """Return the applied allocation for backward compatibility."""
        return self.applied_allocation


@dataclass(frozen=True)
class WalkForwardPortfolioAllocationResult:
    """Fold details and aggregate out-of-sample portfolio results."""

    strategy_names: tuple[str, ...]
    folds: tuple[WalkForwardFold, ...]
    average_allocation: torch.Tensor
    daily_results: pl.DataFrame
    statistics: PortfolioStatistics
    diversification: PortfolioDiversification


def find_optimal_capital_allocation_from_csv(
    portfolio_name: str,
    alpha: float,
    grid_increment: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    bootstrap_on: bool = False,
    bootstrap_runs: int = 1024,
    bootstrap_length: int = 256,
    percentile: float = 10.0,
    max_total_allocation: float = 1.0,
    annualization_periods: int = _ANNUALIZATION_PERIODS,
    risk_free_rate: float = 0.0,
    show_plot: bool = True,
    refinement_runs: int = 0,
    refinement_divisor: float = 2.0,
    ADD_limit: float | None = None,  # pylint: disable=invalid-name
    returns_dtype: type[pl.DataType] = pl.Float32,
) -> PortfolioAllocationResult:
    """Optimize and report allocations for the CSV strategies in a portfolio.

    CSV files are read from
    ``DATA_FOLDER/results/portfolios/{portfolio_name}``. Each file is one
    strategy and must have case-insensitive ``Opened`` and ``ROR`` columns.
    ``Opened`` is reduced to a date, and ``ROR / 100`` is used as the daily
    return on risk. The files must contain no more than one row per date.

    The returned daily frame spans every NYSE session from the first observed
    date through the last. A null strategy value means that strategy was
    inactive on the session; these nulls become NaNs in the optimizer input so
    disjoint strategies are recognized correctly. They are treated as zero
    when calculating the combined portfolio results.

    Win rate is measured over sessions on which at least one strategy with a
    nonzero allocation was active; a zero-return active session remains in its
    denominator. ADD, MaxDD, duration, volatility, and the risk ratios use the
    complete NYSE session series, including inactive zero-return sessions.
    When bootstrapping is enabled, the report also includes the winning
    candidate's bootstrap ADD retained by the optimizer at the chosen percentile.
    Pairwise strategy correlations likewise use the complete series after
    replacing inactive values with zero. Correlations involving a constant
    strategy are reported as undefined and excluded from correlation summaries.

    Args:
        portfolio_name: Name of the directory below ``results/portfolios``.
        alpha: Drawdown-penalty weight passed to the allocation optimizer.
        grid_increment: Requested allocation grid increment.
        device: Optimizer device. Uses the optimizer's CUDA/CPU default when omitted.
        dtype: Output allocation dtype. Defaults to the input-series dtype.
        bootstrap_on: Whether ADD is calculated from bootstrapped paths.
        bootstrap_runs: Number of paths used by bootstrapped ADD.
        bootstrap_length: Fixed number of sessions in each bootstrap path.
        percentile: Pessimistic bootstrap ADD percentile, expressed in percent.
        max_total_allocation: Allocation cap for each overlapping strategy group.
        annualization_periods: Sessions per year for volatility and risk ratios.
        risk_free_rate: Annual effective risk-free rate used by Sharpe and Sortino.
        show_plot: Whether to display the combined log-scale equity curve.
        refinement_runs: Number of local grid refinement levels after the initial
            search. Each level is recentered until its result is unchanged.
        refinement_divisor: Factor dividing the increment at every refinement.
        ADD_limit: Optional inclusive minimum acceptable ADD in ``[-1, 0]``.
            Historical ADD is used unless bootstrapping is enabled.
        returns_dtype: Floating-point dtype used to load and store strategy
            returns. Must be ``polars.Float32`` or ``polars.Float64`` and
            defaults to ``polars.Float32``.

    Returns:
        Strategy names, optimal allocation, combined daily results, financial
        statistics, and diversification statistics.

    Raises:
        FileNotFoundError: If the portfolio directory or its CSV files do not exist.
        TypeError: If a wrapper parameter has an invalid type.
        ValueError: If a CSV is malformed or contains invalid observations.
    """
    annualization_value = _validate_annualization_periods(annualization_periods)
    risk_free_value = _validate_risk_free_rate(risk_free_rate)
    if not isinstance(show_plot, bool):
        raise TypeError("show_plot must be a bool")

    strategy_names, synchronized_returns, returns_tensor = _load_portfolio_returns(
        portfolio_name, returns_dtype
    )

    optimizer_diagnostics = (
        capital_allocation.CapitalAllocationDiagnostics() if bootstrap_on else None
    )
    allocation = capital_allocation.find_optimal_capital_allocation(
        returns_tensor,
        alpha=alpha,
        grid_increment=grid_increment,
        device=device,
        dtype=dtype,
        bootstrap_on=bootstrap_on,
        bootstrap_runs=bootstrap_runs,
        bootstrap_length=bootstrap_length,
        percentile=percentile,
        max_total_allocation=max_total_allocation,
        refinement_runs=refinement_runs,
        refinement_divisor=refinement_divisor,
        diagnostics=optimizer_diagnostics,
        ADD_limit=ADD_limit,
    )
    weights = allocation.detach().to(device="cpu", dtype=torch.float64).tolist()
    daily_results = _combined_daily_results(
        synchronized_returns, strategy_names, weights
    )
    statistics = _portfolio_statistics(
        daily_results,
        annualization_value,
        risk_free_value,
        (
            optimizer_diagnostics.bootstrap_average_drawdown
            if optimizer_diagnostics is not None
            else None
        ),
    )
    diversification = _portfolio_diversification(
        returns_tensor, strategy_names, allocation
    )
    result = PortfolioAllocationResult(
        strategy_names=strategy_names,
        allocation=allocation,
        daily_results=daily_results,
        statistics=statistics,
        diversification=diversification,
    )
    _print_report(portfolio_name, result)
    if show_plot:
        _plot_equity_curve(portfolio_name, daily_results)
    return result


def walk_forward_capital_allocation_from_csv(
    portfolio_name: str,
    alpha: float,
    grid_increment: float,
    training_weeks: int,
    embargo_weeks: int,
    simulation_weeks: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    bootstrap_on: bool = False,
    bootstrap_runs: int = 1024,
    bootstrap_length: int = 256,
    percentile: float = 10.0,
    max_total_allocation: float = 1.0,
    annualization_periods: int = _ANNUALIZATION_PERIODS,
    risk_free_rate: float = 0.0,
    show_plot: bool = True,
    refinement_runs: int = 0,
    refinement_divisor: float = 2.0,
    ADD_limit: float | None = None,  # pylint: disable=invalid-name
    allocation_alpha: float = 1.0,
    returns_dtype: type[pl.DataType] = pl.Float32,
) -> WalkForwardPortfolioAllocationResult:
    """Run rolling portfolio optimization and report out-of-sample results.

    Calendar weeks are Monday-anchored. A partial first week is discarded,
    except when the first date is that week's first NYSE session because of a
    market holiday. Each fold trains over ``training_weeks``, skips
    ``embargo_weeks``, and applies its independently calculated allocation for
    ``simulation_weeks``. The full structure then advances by
    ``simulation_weeks`` until the data ends; the final simulation may be
    partial.

    Every fold invokes :func:`ifera.allocation.find_optimal_capital_allocation`
    from scratch,
    including independent bootstrap samples when enabled. Reported statistics
    are calculated only from the concatenated out-of-sample periods and always
    use historical ADD; bootstrap ADD is not reported.
    ``returns_dtype`` controls the synchronized strategy-return storage and
    optimizer inputs, and defaults to ``polars.Float32``.

    ``allocation_alpha`` optionally smooths changes after the first fold using
    ``applied = allocation_alpha * calculated + (1 - allocation_alpha) *
    previous_applied``. It must be finite and between zero and one, inclusive.
    The optimizer itself always runs independently and is not influenced by the
    smoothed allocation. Smoothing is post-processing, so the applied allocation
    is not projected back into a later fold's potentially changed overlap,
    maximum-allocation, or ``ADD_limit`` constraints.
    """
    from .walk_forward_allocation import (  # pylint: disable=import-outside-toplevel
        walk_forward_capital_allocation_from_csv as run_walk_forward,
    )

    return run_walk_forward(
        portfolio_name=portfolio_name,
        alpha=alpha,
        grid_increment=grid_increment,
        training_weeks=training_weeks,
        embargo_weeks=embargo_weeks,
        simulation_weeks=simulation_weeks,
        device=device,
        dtype=dtype,
        bootstrap_on=bootstrap_on,
        bootstrap_runs=bootstrap_runs,
        bootstrap_length=bootstrap_length,
        percentile=percentile,
        max_total_allocation=max_total_allocation,
        annualization_periods=annualization_periods,
        risk_free_rate=risk_free_rate,
        show_plot=show_plot,
        refinement_runs=refinement_runs,
        refinement_divisor=refinement_divisor,
        ADD_limit=ADD_limit,
        allocation_alpha=allocation_alpha,
        returns_dtype=returns_dtype,
    )


def _validate_annualization_periods(value: int) -> int:
    """Return a validated number of periods per year."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("annualization_periods must be an integer")
    if value <= 0:
        raise ValueError("annualization_periods must be positive")
    return value


def _validate_risk_free_rate(value: float) -> float:
    """Return a validated annual effective risk-free rate."""
    if isinstance(value, bool):
        raise TypeError("risk_free_rate must be a real number")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("risk_free_rate must be a real number") from exc
    if not math.isfinite(rate) or rate <= -1.0:
        raise ValueError("risk_free_rate must be finite and greater than -1.0")
    return rate


def _portfolio_path(portfolio_name: str) -> Path:
    """Resolve a safe portfolio directory below the configured data folder."""
    if not isinstance(portfolio_name, str):
        raise TypeError("portfolio_name must be a string")
    if (
        not portfolio_name
        or portfolio_name in {".", ".."}
        or Path(portfolio_name).name != portfolio_name
        or "/" in portfolio_name
        or "\\" in portfolio_name
    ):
        raise ValueError("portfolio_name must be a single nonempty directory name")
    portfolio_path = (
        Path(settings.DATA_FOLDER) / "results" / "portfolios" / portfolio_name
    )
    if not portfolio_path.is_dir():
        raise FileNotFoundError(f"Portfolio directory does not exist: {portfolio_path}")
    return portfolio_path


def _portfolio_csv_paths(portfolio_path: Path) -> tuple[Path, ...]:
    """Return portfolio CSV paths in deterministic filename order."""
    csv_paths = tuple(
        sorted(
            (
                path
                for path in portfolio_path.iterdir()
                if path.is_file() and path.suffix.casefold() == ".csv"
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )
    if not csv_paths:
        raise ValueError(f"No CSV files found in portfolio: {portfolio_path}")
    return csv_paths


def _validate_strategy_names(strategy_names: tuple[str, ...]) -> None:
    """Reject ambiguous or result-column-conflicting strategy names."""
    normalized = tuple(name.strip().casefold() for name in strategy_names)
    if len(set(normalized)) != len(normalized):
        raise ValueError("CSV filenames must have unique case-insensitive stems")
    conflicts = sorted(set(normalized) & _RESULT_COLUMN_NAMES)
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"CSV filename stems conflict with result columns: {joined}")


def _required_column(columns: list[str], required_name: str, path: Path) -> str:
    """Find exactly one case-insensitive CSV column name."""
    matches = [
        column
        for column in columns
        if column.strip().casefold() == required_name.casefold()
    ]
    if not matches:
        raise ValueError(f"{path.name} is missing the {required_name!r} column")
    if len(matches) != 1:
        raise ValueError(f"{path.name} has multiple columns matching {required_name!r}")
    return matches[0]


def _opened_date_expression(column_name: str) -> pl.Expr:
    """Build an expression accepting the example format and strict ISO variants."""
    opened = pl.col(column_name).cast(pl.String).str.strip_chars()
    parsed_values: list[pl.Expr] = []
    for date_format in _CSV_DATE_FORMATS:
        if "%H" in date_format or "%I" in date_format:
            parsed = opened.str.to_datetime(date_format, strict=False).dt.date()
        else:
            parsed = opened.str.to_date(date_format, strict=False)
        parsed_values.append(parsed)
    return pl.coalesce(parsed_values).alias("date")


def _validate_returns_dtype(value: object) -> type[pl.DataType]:
    """Return a supported floating-point dtype for loaded strategy returns."""
    if value == pl.Float32:
        return pl.Float32
    if value == pl.Float64:
        return pl.Float64
    raise ValueError("returns_dtype must be polars.Float32 or polars.Float64")


def _load_strategy_csv(
    path: Path, returns_dtype: type[pl.DataType] = pl.Float32
) -> pl.DataFrame:
    """Load and validate one strategy's dated fractional daily returns."""
    returns_dtype_value = _validate_returns_dtype(returns_dtype)
    try:
        raw_frame = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"Could not read strategy CSV {path.name}: {exc}") from exc
    if raw_frame.height == 0:
        raise ValueError(f"Strategy CSV is empty: {path.name}")

    opened_column = _required_column(raw_frame.columns, "Opened", path)
    return_column = _required_column(raw_frame.columns, "ROR", path)
    try:
        strategy_frame = raw_frame.select(
            _opened_date_expression(opened_column),
            (
                (pl.col(return_column).cast(pl.Float64, strict=True) / 100.0).cast(
                    returns_dtype_value, strict=True
                )
            ).alias("return"),
        )
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f"Could not parse {path.name}: {exc}") from exc

    if strategy_frame["date"].null_count() != 0:
        raise ValueError(f"{path.name} contains an invalid Opened date")
    invalid_returns = strategy_frame.filter(
        pl.col("return").is_null() | ~pl.col("return").is_finite()
    )
    if invalid_returns.height:
        raise ValueError(f"{path.name} contains a non-finite ROR value")
    duplicate_dates = (
        strategy_frame.group_by("date").len().filter(pl.col("len") > 1)["date"]
    )
    if not duplicate_dates.is_empty():
        dates = ", ".join(str(value) for value in duplicate_dates.sort().to_list())
        raise ValueError(f"{path.name} contains duplicate Opened dates: {dates}")
    return strategy_frame.sort("date")


def _nyse_market_dates(start_date: date, end_date: date) -> pl.DataFrame:
    """Return all full NYSE session dates in an inclusive calendar range."""
    years = range(start_date.year, end_date.year + 1)
    nyse_holidays = NYSE(years=years)
    holiday_dates = pl.Series(  # pylint: disable=assignment-from-no-return
        "holiday_date", sorted(nyse_holidays.keys()), dtype=pl.Date
    ).implode()
    return (
        pl.DataFrame(
            {"date": pl.date_range(start_date, end_date, interval="1d", eager=True)}
        )
        .filter(
            (pl.col("date").dt.weekday() <= 5) & ~pl.col("date").is_in(holiday_dates)
        )
        .sort("date")
    )


def _synchronize_strategy_returns(
    strategy_names: tuple[str, ...], strategy_frames: tuple[pl.DataFrame, ...]
) -> pl.DataFrame:
    """Align strategy returns to every NYSE session in their shared date range."""
    observed_dates = [
        value
        for strategy_frame in strategy_frames
        for value in strategy_frame["date"].to_list()
    ]
    start_date = min(observed_dates)
    end_date = max(observed_dates)
    market_dates = _nyse_market_dates(start_date, end_date)
    valid_dates = set(market_dates["date"].to_list())

    synchronized = market_dates
    for strategy_name, strategy_frame in zip(strategy_names, strategy_frames):
        nonmarket_dates = sorted(set(strategy_frame["date"].to_list()) - valid_dates)
        if nonmarket_dates:
            dates = ", ".join(str(value) for value in nonmarket_dates)
            raise ValueError(
                f"{strategy_name}.csv contains non-NYSE market dates: {dates}"
            )
        synchronized = synchronized.join(
            strategy_frame.rename({"return": strategy_name}),
            on="date",
            how="left",
            validate="1:1",
        )
    return synchronized


def _load_portfolio_returns(
    portfolio_name: str,
    returns_dtype: type[pl.DataType] = pl.Float32,
) -> tuple[tuple[str, ...], pl.DataFrame, torch.Tensor]:
    """Load, validate, and synchronize every strategy in a named portfolio."""
    portfolio_path = _portfolio_path(portfolio_name)
    csv_paths = _portfolio_csv_paths(portfolio_path)
    strategy_names = tuple(path.stem for path in csv_paths)
    _validate_strategy_names(strategy_names)
    strategy_frames = tuple(
        _load_strategy_csv(path, returns_dtype) for path in csv_paths
    )
    synchronized_returns = _synchronize_strategy_returns(
        strategy_names, strategy_frames
    )
    returns_tensor = synchronized_returns.select(strategy_names).to_torch()
    return strategy_names, synchronized_returns, returns_tensor


def _combined_daily_results(
    synchronized_returns: pl.DataFrame,
    strategy_names: tuple[str, ...],
    weights: list[float],
) -> pl.DataFrame:
    """Apply an allocation and add its combined return, equity, and drawdown."""
    return _add_equity_curve(
        _apply_allocation(synchronized_returns, strategy_names, weights)
    )


def _apply_allocation(
    synchronized_returns: pl.DataFrame,
    strategy_names: tuple[str, ...],
    weights: list[float],
) -> pl.DataFrame:
    """Apply an allocation without resetting or calculating an equity curve."""
    weighted_returns = [
        pl.col(strategy_name).fill_null(0.0) * weight
        for strategy_name, weight in zip(strategy_names, weights)
    ]
    allocated_activity = [
        pl.col(strategy_name).is_not_null()
        for strategy_name, weight in zip(strategy_names, weights)
        if weight != 0.0
    ]
    active = (
        pl.any_horizontal(allocated_activity) if allocated_activity else pl.lit(False)
    )
    combined = synchronized_returns.with_columns(
        active.alias("active"),
        pl.sum_horizontal(weighted_returns).alias("combined_return"),
    )
    invalid = combined.filter(
        ~pl.col("combined_return").is_finite() | (pl.col("combined_return") <= -1.0)
    )
    if invalid.height:
        raise ValueError("Optimal allocation produced an invalid combined daily return")

    return combined


def _add_equity_curve(combined_returns: pl.DataFrame) -> pl.DataFrame:
    """Add one continuous log-equity and drawdown path to daily returns."""
    combined = combined_returns.with_columns(
        pl.col("combined_return").log1p().cum_sum().alias("log_equity")
    )

    running_peak = pl.max_horizontal(pl.col("log_equity").cum_max(), pl.lit(0.0))
    return combined.with_columns(
        pl.col("log_equity").exp().alias("equity"),
        (pl.col("log_equity") - running_peak).exp().sub(1.0).alias("drawdown"),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide, returning a signed infinity when a nonzero numerator has no risk."""
    if denominator > 0.0:
        return numerator / denominator
    if numerator > 0.0:
        return math.inf
    if numerator < 0.0:
        return -math.inf
    return 0.0


def _max_drawdown_duration(daily_results: pl.DataFrame) -> int:
    """Return the longest run of consecutive NYSE sessions below a prior peak."""
    underwater_runs = (
        daily_results.select(
            pl.col("drawdown"),
            (pl.col("drawdown") >= 0.0).cum_sum().alias("peak_group"),
        )
        .filter(pl.col("drawdown") < 0.0)
        .group_by("peak_group")
        .len()
    )
    if underwater_runs.is_empty():
        return 0
    maximum = underwater_runs["len"].max()
    if not isinstance(maximum, int):
        raise RuntimeError("Could not calculate maximum drawdown duration")
    return maximum


def _date_aware_compound_growth_rate(
    start_date: date,
    end_date: date,
    ending_log_equity: float,
    periods_per_year: int,
) -> float:
    """Calculate a compound growth rate from inclusive duration and log equity."""
    years = ((end_date - start_date).days + 1) / 365.25
    exponent = ending_log_equity / (years * periods_per_year)
    try:
        return math.expm1(exponent)
    except OverflowError:
        return math.inf


def _portfolio_statistics(
    daily_results: pl.DataFrame,
    annualization_periods: int,
    risk_free_rate: float,
    bootstrap_average_drawdown: float | None,
) -> PortfolioStatistics:
    """Calculate portfolio statistics from a complete NYSE daily series."""
    daily_risk_free = math.expm1(math.log1p(risk_free_rate) / annualization_periods)
    returns = pl.col("combined_return")
    active_returns = returns.filter(pl.col("active"))
    excess_returns = returns - daily_risk_free
    downside_returns = pl.when(excess_returns < 0.0).then(excess_returns).otherwise(0.0)
    summary = daily_results.select(
        pl.len().alias("market_days"),
        pl.col("active").sum().alias("active_days"),
        (returns > 0.0).sum().alias("winning_days"),
        (returns < 0.0).sum().alias("losing_days"),
        returns.mean().alias("average_daily_return"),
        returns.std(ddof=1).alias("daily_volatility"),
        excess_returns.mean().alias("average_excess_return"),
        downside_returns.pow(2).mean().sqrt().alias("downside_deviation"),
        returns.filter(returns > 0.0).sum().alias("gross_profit"),
        returns.filter(returns < 0.0).sum().abs().alias("gross_loss"),
        active_returns.max().alias("best_day"),
        active_returns.min().alias("worst_day"),
        pl.col("equity").last().alias("ending_equity"),
        pl.col("log_equity").last().alias("ending_log_equity"),
        pl.col("drawdown").pow(2).mean().sqrt().neg().alias("average_drawdown"),
        pl.col("drawdown").min().alias("max_drawdown"),
    ).row(0, named=True)

    market_days = int(summary["market_days"])
    active_days = int(summary["active_days"])
    ending_equity = float(summary["ending_equity"])
    daily_volatility = float(summary["daily_volatility"] or 0.0)
    average_excess_return = float(summary["average_excess_return"])
    downside_deviation = float(summary["downside_deviation"])
    start_date = daily_results["date"].item(0)
    end_date = daily_results["date"].item(-1)
    ending_log_equity = float(summary["ending_log_equity"])
    cagr = _date_aware_compound_growth_rate(
        start_date, end_date, ending_log_equity, periods_per_year=1
    )
    cmgr = _date_aware_compound_growth_rate(
        start_date, end_date, ending_log_equity, periods_per_year=12
    )
    annualization_scale = math.sqrt(annualization_periods)
    max_drawdown = float(summary["max_drawdown"])
    gross_profit = float(summary["gross_profit"] or 0.0)
    gross_loss = float(summary["gross_loss"] or 0.0)
    return PortfolioStatistics(
        start_date=start_date,
        end_date=end_date,
        market_days=market_days,
        active_days=active_days,
        winning_days=int(summary["winning_days"]),
        losing_days=int(summary["losing_days"]),
        ending_equity=ending_equity,
        total_return=ending_equity - 1.0,
        cagr=cagr,
        cmgr=cmgr,
        win_rate=(int(summary["winning_days"]) / active_days if active_days else 0.0),
        average_drawdown=float(summary["average_drawdown"]),
        bootstrap_average_drawdown=bootstrap_average_drawdown,
        max_drawdown=max_drawdown,
        max_drawdown_duration=_max_drawdown_duration(daily_results),
        average_daily_return=float(summary["average_daily_return"]),
        annualized_volatility=daily_volatility * annualization_scale,
        sharpe_ratio=_safe_ratio(average_excess_return, daily_volatility)
        * annualization_scale,
        sortino_ratio=_safe_ratio(average_excess_return, downside_deviation)
        * annualization_scale,
        calmar_ratio=_safe_ratio(cagr, abs(max_drawdown)),
        profit_factor=_safe_ratio(gross_profit, gross_loss),
        exposure=active_days / market_days,
        best_day=float(summary["best_day"] or 0.0),
        worst_day=float(summary["worst_day"] or 0.0),
    )


def _strategy_correlation_matrix(returns: torch.Tensor) -> torch.Tensor:
    """Return a CPU float64 Pearson matrix with inactive NaNs replaced by zero."""
    filled_returns = torch.nan_to_num(
        returns.detach().to(device="cpu", dtype=torch.float64), nan=0.0
    )
    column_scales = filled_returns.abs().amax(dim=0)
    safe_scales = torch.where(
        column_scales > 0.0, column_scales, torch.ones_like(column_scales)
    )
    scaled_returns = filled_returns / safe_scales
    centered_returns = scaled_returns - scaled_returns.mean(dim=0, keepdim=True)
    norms = centered_returns.square().sum(dim=0).sqrt()
    cross_products = centered_returns.transpose(0, 1) @ centered_returns
    denominators = torch.outer(norms, norms)
    correlation_matrix = torch.full_like(cross_products, math.nan)
    nonconstant = torch.amax(scaled_returns, dim=0) > torch.amin(scaled_returns, dim=0)
    defined = (denominators > 0.0) & torch.outer(nonconstant, nonconstant)
    correlation_matrix[defined] = cross_products[defined] / denominators[defined]
    correlation_matrix.clamp_(-1.0, 1.0)
    correlation_matrix[correlation_matrix.abs() <= _CORRELATION_ZERO_TOLERANCE] = 0.0
    return correlation_matrix


def _valid_pairwise_correlations(
    correlation_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return finite unique-pair correlations, their indices, and total count."""
    strategy_count = correlation_matrix.shape[0]
    pair_indices = torch.triu_indices(strategy_count, strategy_count, offset=1)
    pair_correlations = correlation_matrix[pair_indices[0], pair_indices[1]]
    valid = torch.isfinite(pair_correlations)
    return pair_correlations[valid], pair_indices[:, valid], pair_indices.shape[1]


def _selected_correlation(
    correlations: torch.Tensor,
    pair_indices: torch.Tensor,
    strategy_names: tuple[str, ...],
    *,
    minimum: bool,
) -> tuple[float | None, tuple[str, str] | None]:
    """Return a selected correlation and its deterministically ordered pair."""
    if correlations.numel() == 0:
        return None, None
    selected = torch.argmin(correlations) if minimum else torch.argmax(correlations)
    selected_index = int(selected.item())
    first = int(pair_indices[0, selected_index].item())
    second = int(pair_indices[1, selected_index].item())
    return float(correlations[selected_index].item()), (
        strategy_names[first],
        strategy_names[second],
    )


def _allocation_concentration(
    allocation: torch.Tensor,
) -> tuple[int, float | None, float]:
    """Return allocated count, normalized-weight HHI, and effective count."""
    weights = allocation.detach().to(device="cpu", dtype=torch.float64)
    allocated_strategy_count = int(torch.count_nonzero(weights > 0.0).item())
    total_weight = float(weights.sum().item())
    if total_weight <= 0.0:
        return allocated_strategy_count, None, 0.0
    normalized_weights = weights / total_weight
    concentration = float(normalized_weights.square().sum().item())
    return allocated_strategy_count, concentration, 1.0 / concentration


def _portfolio_diversification(
    returns: torch.Tensor,
    strategy_names: tuple[str, ...],
    allocation: torch.Tensor,
) -> PortfolioDiversification:
    """Calculate correlation and allocation-concentration summaries."""
    correlation_matrix = _strategy_correlation_matrix(returns)
    correlations, pair_indices, total_pair_count = _valid_pairwise_correlations(
        correlation_matrix
    )
    highest_correlation, highest_pair = _selected_correlation(
        correlations, pair_indices, strategy_names, minimum=False
    )
    negative = correlations < 0.0
    lowest_negative_correlation, lowest_negative_pair = _selected_correlation(
        correlations[negative], pair_indices[:, negative], strategy_names, minimum=True
    )
    allocated_count, concentration, effective_count = _allocation_concentration(
        allocation
    )
    return PortfolioDiversification(
        correlation_matrix=correlation_matrix,
        strategy_count=len(strategy_names),
        allocated_strategy_count=allocated_count,
        total_pair_count=total_pair_count,
        valid_pair_count=correlations.numel(),
        highest_correlation=highest_correlation,
        highest_correlation_pair=highest_pair,
        lowest_negative_correlation=lowest_negative_correlation,
        lowest_negative_correlation_pair=lowest_negative_pair,
        mean_absolute_correlation=(
            float(correlations.abs().mean().item()) if correlations.numel() else None
        ),
        mean_correlation=(
            float(correlations.mean().item()) if correlations.numel() else None
        ),
        allocation_concentration=concentration,
        effective_strategy_count=effective_count,
    )


def _format_percentage(value: float) -> str:
    """Format a fraction as a percentage for console output."""
    return f"{value:.2%}"


def _format_ratio(value: float) -> str:
    """Format a finite or infinite financial ratio."""
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:.3f}"


def _format_optional_ratio(value: float | None) -> str:
    """Format an optional ratio, using N/A when it is undefined."""
    return "N/A" if value is None else _format_ratio(value)


def _format_correlation_pair(
    value: float | None, strategy_pair: tuple[str, str] | None
) -> str:
    """Format a correlation together with the strategy pair that produced it."""
    if value is None or strategy_pair is None:
        return "N/A"
    return f"{_format_ratio(value)} ({strategy_pair[0]}, {strategy_pair[1]})"


def _print_report(portfolio_name: str, result: PortfolioAllocationResult) -> None:
    """Print allocations and combined financial statistics."""
    statistics = result.statistics
    print(f"\nOptimal capital allocation: {portfolio_name}")
    print(f"Period: {statistics.start_date} to {statistics.end_date}")
    print("Allocations:")
    weights = result.allocation.detach().to(device="cpu", dtype=torch.float64).tolist()
    for strategy_name, weight in zip(result.strategy_names, weights):
        print(f"  {strategy_name}: {_format_percentage(weight)}")
    print(f"  Total: {_format_percentage(sum(weights))}")

    _print_combined_statistics(statistics)
    _print_diversification(result.diversification)


def _print_combined_statistics(statistics: PortfolioStatistics) -> None:
    """Print combined portfolio performance statistics."""

    print("Combined portfolio statistics:")
    print(f"  Ending equity: {statistics.ending_equity:.4f}")
    print(f"  Total return: {_format_percentage(statistics.total_return)}")
    print(f"  CAGR: {_format_percentage(statistics.cagr)}")
    print(f"  CMGR: {_format_percentage(statistics.cmgr)}")
    print(f"  Win rate: {_format_percentage(statistics.win_rate)}")
    print(f"  ADD: {_format_percentage(statistics.average_drawdown)}")
    if statistics.bootstrap_average_drawdown is not None:
        print(
            "  ADD (bootstrap): "
            f"{_format_percentage(statistics.bootstrap_average_drawdown)}"
        )
    print(f"  MaxDD: {_format_percentage(statistics.max_drawdown)}")
    print(f"  MaxDD duration: {statistics.max_drawdown_duration} market days")
    print(f"  Sharpe ratio: {_format_ratio(statistics.sharpe_ratio)}")
    print(f"  Sortino ratio: {_format_ratio(statistics.sortino_ratio)}")
    print(f"  Calmar ratio: {_format_ratio(statistics.calmar_ratio)}")
    print(f"  Profit factor: {_format_ratio(statistics.profit_factor)}")
    print(
        "  Annualized volatility: "
        f"{_format_percentage(statistics.annualized_volatility)}"
    )
    print(
        "  Average market-day return: "
        f"{_format_percentage(statistics.average_daily_return)}"
    )
    print(f"  Exposure: {_format_percentage(statistics.exposure)}")
    print(f"  Active days: {statistics.active_days}/{statistics.market_days}")
    print(f"  Best day: {_format_percentage(statistics.best_day)}")
    print(f"  Worst day: {_format_percentage(statistics.worst_day)}")


def _print_diversification(diversification: PortfolioDiversification) -> None:
    """Print portfolio robustness and diversification statistics."""
    print("\nPortfolio robustness / diversification:")
    print(
        "  Highest pairwise correlation: "
        + _format_correlation_pair(
            diversification.highest_correlation,
            diversification.highest_correlation_pair,
        )
    )
    print(
        "  Lowest negative correlation: "
        + _format_correlation_pair(
            diversification.lowest_negative_correlation,
            diversification.lowest_negative_correlation_pair,
        )
    )
    print(
        "  Mean absolute correlation: "
        + _format_optional_ratio(diversification.mean_absolute_correlation)
    )
    print(
        "  Mean pairwise correlation: "
        + _format_optional_ratio(diversification.mean_correlation)
    )
    print(
        "  Valid pairwise correlations: "
        f"{diversification.valid_pair_count}/{diversification.total_pair_count}"
    )
    print(
        "  Allocated strategies: "
        f"{diversification.allocated_strategy_count}/{diversification.strategy_count}"
    )
    print(
        "  Allocation concentration (HHI): "
        + _format_optional_ratio(diversification.allocation_concentration)
    )
    print(f"  Effective strategy count: {diversification.effective_strategy_count:.3f}")


def _plot_equity_curve(
    portfolio_name: str,
    daily_results: pl.DataFrame,
    title_prefix: str = "Optimized equity curve",
) -> None:
    """Display the optimized portfolio equity curve on a logarithmic scale."""
    # Imported lazily so non-plotting callers do not pay matplotlib's import cost.
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    _, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        daily_results["date"].to_list(),
        daily_results["equity"].to_list(),
        linewidth=1.5,
    )
    axis.set_yscale("log")
    axis.set_title(f"{title_prefix}: {portfolio_name}")
    axis.set_xlabel("Date")
    axis.set_ylabel("Equity (log scale)")
    axis.grid(visible=True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
