# Strategy optimization

`ifera.strategy_optimization` reads one strategy from
`DATA_FOLDER/results/portfolios/<portfolio>/<strategy>.csv`. It supports one
same-day, 0DTE trade per market day, with the allocation package's option leg
syntax and CSV timestamps. The original CSV is retained; exports go under
`DATA_FOLDER/results/strategy_optimization/<portfolio>/` by default.

Configure the SQLAlchemy Oracle connection through environment variables or `.env`:

```dotenv
IFERA_DB_USER=iferaro
IFERA_DB_PASSWORD=iferaro
IFERA_DSN=IFERA1
```

These are the supplied local, read-only development credentials. Other databases
should supply their own credentials. Oracle Net configuration must resolve the
DSN, or it can be an Easy Connect string such as `host:1521/service`.

```python
from datetime import date

from ifera.strategy_optimization import (
    optimize_strategy_from_csv,
    return_over_add,
)

result = optimize_strategy_from_csv(
    "pf_005_exp",
    "AftPC2_raw",
    mode="combined",  # entry_filter, early_exit, or combined
    metric=return_over_add,
    optimizer_parameters={
        "max_conjunctions": 5,
        "max_population": 256,
        "generations": 100,
        "seed": 42,
        "objective_batch_size": 16,
    },
    exit_slippage=0.05,
    # start_date=date(2023, 3, 1),
    # end_date=date(2026, 11, 12),
    # features_file="my_features.yaml",
    # output_csv="optimized.csv",
)

print(result.conditions)
print(result.statistics)
print(result.conditions_yaml)
```

Constructor hyperparameters and `max_population`/`generations` belong in
`optimizer_parameters`; feature counts are inferred. The returned result includes
the optimizer result tensors, selected features, winning trades, complete daily
calendar, allocation-adjusted statistics, capital allocation fraction, objective
and fitness histories, export path, and plot.
Use `show_plot=False`, `show_progress=False`, or `print_report=False` for batch
runs. `market_data_source` can supply a caller-owned `OracleMarketDataSource`
with a SQLAlchemy engine, or an offline object implementing
`load(StrategyData, lookback_days=400) -> MarketData`.

Use inclusive `start_date` and `end_date` to select a window covered by Oracle.
Dates are never silently dropped: the selected calendar retains no-trade
sessions, and missing trade-minute data raises an error. The indicator lookback
extends before the selected start and may precede Oracle's loaded history;
missing daily bars can be supplied by Yahoo as described below.

## Saved conditions and replay

Every optimization saves `<output stem>.conditions.yaml` beside its modified
CSV. `result.conditions_yaml` returns this path. The sidecar contains the final
simplified entry filter and/or early-exit rule, the resolved definitions of
referenced features, exit slippage, indicator lookback, and the actual selected
date window, including idle market sessions at its boundaries.

Use the saved conditions to recreate a modified CSV without running evolution:

```python
from ifera.strategy_optimization import apply_strategy_conditions

recreated_csv = apply_strategy_conditions(
    raw_csv="data/results/portfolios/pf_005_exp/AftPC2_raw.csv",
    conditions_yaml=result.conditions_yaml,
    output_csv="recreated.csv",  # optional
    # market_data_source=offline_source,
    # device="cpu",
)
```

The helper returns the output `Path`. Its default destination is
`<output stem>_recreated.csv` beside the YAML. It accepts any raw CSV path; the
source does not have to be inside a portfolio directory. The original CSV and
condition YAML are protected from overwrite. Column names, order, and extra
ledger fields are preserved using the same accounting/export path as optimization.
Replay does not run a metric, simplification, capital allocation, or plotting.

The versioned YAML has separate `entry_filter` and `early_exit` clause lists:
an empty list is FALSE and a list containing an empty clause is TRUE. Only active,
applicable restrictions are saved. Boolean false literals remain explicit.
Continuous limits are authoritative normalized float32 values, marked by
`threshold_units: normalized`, so saving/loading cannot change an inclusive
threshold through display rounding. `human_readable` contains the corresponding
conditions in original units for inspection.

Feature definitions are snapshots: replay does not reload the current feature
templates or default feature list. It honors the saved date window, lookback,
normalization, granularity, and whole-combination slippage. Referenced features
still use their registered calculator code and underlying/option data, including
daily-history and event caches. Exact CSV reproduction therefore requires the
same original ledger, calculator behavior, and market/event data. A caller-owned
`market_data_source` works as in optimization.

## Conditions and accounting

An active conjunction ANDs its restrictions; conjunctions are ORed. Inactive
conjunctions are false and unrestricted features are omitted. An active clause
with no applicable restrictions is true. Combined mode assigns the first
`ceil(max_conjunctions / 2)` slots to entry and the rest to exit. Category masks
exclude entry-only features from exit evaluation and exit-only features from
entry evaluation. A single combined conjunction therefore provides an entry
filter with no early-exit clauses.

Entry is evaluated at its minute snapshot. Exits are evaluated at each whole
minute strictly after entry and strictly before the recorded close; the first
matching minute wins. Complete simultaneous leg quotes are required for exit
marks. Missing underlying/option minutes raise an error rather than shifting
the exit to a later available quote. An absent leg's delta restriction is
ignored for that trade. Missing data for an existing leg or an indicator fails
restricted conditions; unrestricted conditions still pass.

Option marks use the signed sum of all long leg midpoints minus short leg
midpoints. Slippage is an adverse **whole-combination price-point amount**:
`P/L = (signed_combo_mid - slippage - signed_entry_debit) * 100 * Quantity`.
For a credit spread opened at 2.00, closed at 1.00, and slippage 0.05, the
one-combo profit is $95. Slippage is applied once per combination, irrespective
of leg count. Original closes retain their recorded accounting and status.
CSV `Risk`, `Premium`, and `P/L` are position totals; `Max Profit`/`Max Loss`
are per-combination values in the source format. Repriced exports update Closed,
Status, P/L, ROR, closing prices, underlying at close, and Return % when present,
while preserving all fields and their order.

## Features

[`feature_templates.yaml`](feature_templates.yaml) owns the calculator metadata:
type, category (`entry`, `exit`, `any`), normalization bounds, optional continuous
`granularity`, and required calculator parameters.
[`default_features.yaml`](default_features.yaml) lists the 55 default feature
instances. A custom features file contains only the `features` mapping; it may
select a subset or add instances with different calculator parameters:

```yaml
features:
  is_monday:
    template: is_day
    day: Monday
  rsi_14_intra:
    template: rsi
    days: 14
    days_ago: 0
  return_on_risk:
    template: ror_pct
```

Template metadata is package-owned and cannot be overridden by a custom features
file. Boolean features are placed first automatically; continuous observations
are clipped and normalized to [0, 1] without rounding. Adding a calculator
involves registering it in `_features.py` and declaring its metadata in
`feature_templates.yaml`. No code is evaluated from YAML.

Continuous templates may declare a positive, finite `granularity` in original
feature units. Thresholds are rounded to the nearest multiple of this value,
using ties-to-even, and clipped to the template's minimum/maximum before entry
or exit decisions and objective calculation. Complexity penalties, returned
winning bounds, and condition printouts use the same effective rounded rules.
Exact normalized 0/1 endpoints retain their unrestricted-tail meaning, including
when a template's minimum or maximum is not itself a grid point. Observations
are never rounded. Omitting `granularity` or setting it to `null` disables
threshold rounding; Boolean templates do not accept it.

The packaged defaults use 60 seconds for time, one dollar for premium and risk,
one minute for minutes since entry, and 0.01 in the respective units for all
other continuous templates. For example, a time threshold of 10:14:23 becomes
10:14:00 and an RSI threshold of 55.123 becomes 55.12. Threshold printouts retain
the configured precision, including fractional seconds when needed.

Indicators use regular-session daily OHLC history with the current session
replaced by its OHLC **through the observation minute**. `days_ago=0` uses this
partial session; a positive value uses completed market sessions. RSI(14) uses
exactly 14 closing prices (13 previous closes and the current price) and mean
gains/losses over those 13 changes. Stochastic defaults to slow D(14,3,3), MACD
to histogram(13,26,9); their components are configurable. ADX uses Wilder
smoothing. Momentum is a point change. Time uses seconds since midnight.
Returns, changes, gap, and reward/risk use percentage points; premium and risk
use account dollars. Leg deltas use the quoted option delta, summed if multiple
legs share the same direction/right. Timestamps are naive New York market time.
Session open and previous close use the first and last available regular-session
underlying snapshots. The inspected development feed runs from 09:31 to 16:00.

Moving-average Boolean templates are `underlying_above_ma(days, ma_type)` and
`ma_comparison(days1, days2, ma_type)`. Supported types are SMA, EMA, TRIMA, and
KAMA. The defaults compare the underlying with 5/10/20-day EMAs and compare
EMA(5) with EMA(10) and EMA(20). These are daily averages: every minute advances
the same completed-session history independently, rather than adding each minute
as another day. EMA uses an SMA seed; TRIMA uses symmetric triangular weights;
KAMA uses 2/30 smoothing and needs one more closing price than its period.

Additional oscillators are Stochastic RSI(14,14,3,3), Williams %R(14), Ultimate
(7,14,28), MFI(14), and Chande momentum(9). Each supports `days_ago`. Stochastic
RSI uses the package's existing exact-price RSI convention, then stochastic
scaling and simple K/D smoothing; the default slow D needs 31 price bars.
Chande uses nine consecutive close changes, requiring ten prices. The Boolean
`macd_above_signal(12,26,9,days_ago)` tests the histogram against zero. Missing
indicator warmup stays unknown rather than becoming a false Boolean value.
See the primary implementations for [Stochastic RSI composition](https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_STOCHRSI.c),
[Ultimate](https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ULTOSC.c),
and [KAMA](https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_KAMA.c).

MFI requires genuine underlying volume. The Oracle Greeks feed supplies none,
so the default intraday MFI remains unavailable with that source. A custom
`MarketData` source can supply incremental minute `underlying.volume` and
completed-session `daily_history.volume`; current volume is accumulated only
through the observation minute. Missing volume is never replaced with option
volume, a constant, or the current day's finalized total. This follows the
[money-flow calculation's volume requirement](https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MFI.c).

`return_pct` and `ror_pct` accept `component: last`, `high`, or `low`. Omitting
it retains the existing `last` behavior. High/low include the entry mark and all
marks through the current observation, with the same slippage and position
accounting as the last return. All three components are exit-only.
`position_bid_ask_spread` is the whole combination's ask minus bid in option
price points, independent of position quantity, and is available at entry/exit.
The default normalization range is 0–10 points.

`position_itm_pct` is exit-only and uses option strikes: calls contribute
`100 * (spot - strike) / strike` and puts contribute
`100 * (strike - spot) / strike`. The feature takes the maximum across legs,
regardless of long/short direction. Positive means at least one option is in the
money; negative means every leg is out of the money. Its default range is
[-10, 10] percentage points. This measures option moneyness rather than P/L.

### Event calendars

`event_day(event_type, days)` shifts the observation date by `days` NYSE market
sessions and checks whether the target date is an event. Offsets must be integers
in [-10, 10]: negative looks behind, positive ahead, and zero means today. Weekends
and NYSE holidays never count as sessions, including across year boundaries.
Actual economic releases on a closed market day are not moved to another day.

Defaults include all 13 event types at offset zero. FOMC means the scheduled
meeting's final policy-decision day. Consumer Sentiment includes Michigan's
preliminary and final releases; PMI means the US ISM manufacturing/services
reports. Monthly Expiration is the third Friday, Triple Witching is its quarterly
subset, and First Weekly is the first Friday options expiration each month.
Expiration holidays roll to the preceding market session. End of Month and End
of Quarter mean the final market session in their respective periods.

Published dates come from the [Federal Reserve](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm),
[BLS calendars](https://www.bls.gov/schedule/),
[BEA](https://www.bea.gov/news/schedule/full),
[ISM](https://www.ismworld.org/supply-management-news-and-reports/reports/rob-report-calendar),
and [Michigan release-date documents](https://data.sca.isr.umich.edu/technical-docs.php).
Yearly CSVs live under `DATA_FOLDER/cache/strategy_optimization/events/<event>/<year>.csv`.
They include complete-year coverage metadata, source URLs, and a fetch timestamp;
warm caches are reused and refresh after a year, rather than every day. Event
dates and flags are prepared once before trade-minute feature calculation and
never fetched from the objective function.

When an agency page is unavailable, recent BLS/BEA dates can come from the
[Fed's published release calendar](https://fred.stlouisfed.org/releases/calendar).
Older BLS/BEA fallbacks use [ALFRED release-date archives](https://alfred.stlouisfed.org/release/downloaddates?ff=txt&rid=10):
these can include benchmark/revision releases, and ALFRED may use first data
availability when the publisher supplied no release date. They therefore have
broader coverage than a regular monthly-news calendar; cache provenance retains
the actual source. Historical ISM dates can also come from
[ISM's own press-release archive](https://www.prnewswire.com/news/institute-for-supply-management/),
with a complete-year [recorded-calendar fallback](https://makkler.com/labs/economic-calendar/us-ism-manufacturing).
Monthly PMI fallbacks must cover all twelve release months; dates are not
inferred from a weekday rule.

The loader validates coverage before caching: blocked or incomplete sources raise
an error instead of silently treating missing dates as no events. A completed
year can also be supplied using `EventCalendar.store(EventYear(...))` with explicit
provenance; these classes are exported from `ifera.strategy_optimization`.
Cached schedules reflect their published source at fetch time; they
do not reconstruct historical vintages of postponement announcements.

## Missing daily history

When Oracle lacks a complete historical session needed for indicator lookback,
the loader downloads daily OHLC with `yfinance`. It checks NYSE market dates and
early closes: the ThetaData feed normally has 390 snapshots from 09:31 through
16:00, or 210 through 13:00 on half days. Complete Oracle sessions remain the
preferred source. Only absent or incomplete historical sessions are downloaded,
batched by underlying; SPX maps to Yahoo's S&P 500 price index, `^GSPC`.

Downloaded bars are cached in
`DATA_FOLDER/cache/strategy_optimization/daily/<underlying>.csv`, for example
`data/cache/strategy_optimization/daily/SPX.csv`. Each CSV includes the underlying,
Yahoo ticker, and price-basis version alongside date and OHLC. Subsequent runs
reuse cached dates without contacting Yahoo; incremental requests retain existing
rows, and cache files are replaced atomically. Invalid or incompatible caches
raise an explicit error rather than mixing price bases.

The fallback preserves nominal, as-traded price units. Yahoo's
[`auto_adjust=False` OHLC is still split-adjusted](https://ranaroussi.github.io/yfinance/advanced/price_adjustment.html),
so the downloader includes subsequent split events through the present and undoes
their effect on earlier prices. This includes splits after the requested history
window and reverse splits; the split's effective date is already in its new
price units. Dividend adjustment is disabled and `Adj Close` is not used. The
cache stores the converted nominal bars and never adjusts them a second time.

From the contemporaneous underlying prices described by
[ThetaData's Greeks documentation](https://thetadata.net/docs/operations/option_history_greeks_all.html),
we infer that nominal prices are the compatible historical basis. This does not
make the two vendors' observations identical: ThetaData may sample
bid/ask midpoints, while Yahoo supplies daily trade OHLC. Prices are not rescaled
to conceal those differences.

Daily fallback bars only feed completed-session history and previous close.
Entry/exit minute prices and current-day partial OHLC still come from Oracle;
finalized daily bars never supply a current day's high, low, close, or fabricated
minute observations. Missing option quotes and required trade-minute prices
continue to raise errors. Current or future daily sessions are not cached.

The default facade skips Yahoo completion when the selected features need no
daily history. `OracleMarketDataSource(fill_daily_history=False)` disables it
explicitly for direct loader calls. Custom market-data sources skip Yahoo
completion and can provide replacement bars using `MarketData.daily_history`.
Event features use their separate yearly cache even with a custom market-data
source; populate that cache or select features without events for an offline run.

## Metrics and execution

Custom metrics receive a PyTorch tensor `(candidates, market_days)` in **percent**
RoR, including zero for no-trade sessions. They must return a finite floating
`(candidates,)` tensor on the input device, independently of candidate ordering
or batch size. Values are clamped to at least `1e-6`.

The built-in metrics add 1 and clamp positive:

- `annualized_return`: `sum(RoR%) * 252 / market_days`.
- `sortino`: annualized return divided by annualized downside RMS deviation.
- `return_over_add`: annualized return divided by percentage RMS drawdown,
  reusing allocation's ADD implementation.

For Sortino and return/ADD, negative numerators are left undivided. A tiny
denominator floor keeps no-downside results finite. Compounded equity treats
losses of at least 100% as absorbing bankruptcy; arithmetic metrics retain
actual losses, including slippage beyond original risk. Statistics reuse the
allocation package. Metrics expose an `annualization_periods` keyword for use
with `functools.partial` if needed.

Use `with_scarcity_penalty` to discourage metrics based on too few trades:

```python
from ifera.strategy_optimization import return_over_add, with_scarcity_penalty

metric = with_scarcity_penalty(
    return_over_add, lower_threshold=0.1, upper_threshold=0.4
)
# Pass metric=metric to optimize_strategy_from_csv(...).
```

The wrapper evaluates the child metric once, then calculates the fraction of
market days with nonzero returns, counting both gains and losses. Below the
lower threshold it returns `1e-6`; at or above the upper threshold it preserves
the child score. Between them, it multiplies by
`sqrt(1e-6 + (1 - 1e-6) * (frequency - lower) / (upper - lower))`.
Thresholds must satisfy `0 <= lower < upper <= 1`. At the lower threshold the
factor is `sqrt(1e-6)`, as distinct from the hard cutoff below it. The optimizer
applies its usual objective floor after the wrapper. With only daily returns
as input, breakeven trades cannot be distinguished from no-trade days and count
as inactive for this penalty.

## Final rule simplification

Strategy optimization always enables `simplify=True`, including when
`optimizer_parameters` requests `simplify=False`. The standalone
`EvolutionaryDNFOptimizer` keeps its default of `simplify=False`.

After the final generation, the winning thresholds are rounded to their
configured grids. The optimizer tests removing one Boolean literal or one
continuous inequality at a time: a lower bound becomes 0, or an upper bound
becomes 1, in normalized units. It accepts the removal with the highest strictly
improved fitness, then repeats until no removal improves fitness. Equal-fitness
changes are rejected. Active conjunctions are never deleted, deactivated,
reordered, or moved between entry and exit; removing a clause's final restriction
leaves an active `TRUE` clause.

Each trial uses the same metric, rounded rule evaluation, and Boolean/continuous
complexity penalties as evolution. Removing a restriction can therefore improve
penalized fitness while reducing the raw objective. With zero complexity
penalties, a redundant restriction with an unchanged objective is retained.
Conditions, exported trades, allocation, and statistics all use the refined
winner.

`objective_history` and `fitness_history` retain `generations + 1` points. The
final point is replaced with the refined winner's objective and fitness; earlier
points describe the generation callbacks. The objective graph may consequently
fall at its last point while fitness improves. The optimizer result exposes
`simplification_steps` for accepted removals and `simplification_evaluations` for
scored trial candidates; total `evaluations` includes these additional trials.

## Final capital allocation and reporting

After selecting the winning strategy, the optimizer calls the existing capital
allocation engine with that strategy as a single-strategy portfolio. Settings are
`alpha=0`, `grid_increment=0.001`, `bootstrap_on=False`, `refinement_runs=0`, and
`ADD_limit=-0.1`. This calculation runs once and uses the full original market
calendar, including sessions excluded by the entry filter.

The printed report includes the chosen capital fraction. `result.allocation`
stores that fraction; `result.statistics` and `result.daily_results["combined_return"]`
describe the funded portfolio. The daily frame's `strategy_return` column retains
unallocated fractional trade returns, with nulls on no-trade days. An all-cash
allocation has zero active days and flat equity.

Evolutionary objectives continue to receive percentage RoR on original trade
risk. The optimized CSV retains the winning trades' original-schema P/L and
RoR accounting, including recalculated early exits. Capital allocation changes
the final portfolio report and daily equity path.

Oracle loading uses the inspected 21c schema and fixed SQL in
[`_oracle_queries.py`](_oracle_queries.py). A load executes two bulk queries,
independent of the number of trades or underlying symbols. Both queries join
`OPT_SYMBOL` to map CSV symbols to option roots, including SPX to SPXW.

Underlying history uses provider sessions from `OPT_MARKET_DATE_MV` and one
positive `UNDERLYING_PRICE` observation per minute from
`TDO_GREEKS_ALL_1M_RAW`. The spot price is repeated across option contracts, so
an indexed lookup with a stop-key avoids scanning and averaging those copies.
Option data is loaded for all requested legs together, generating only their
entry-to-close minutes and using the full primary key
`(SYMBOL, DATETIME, EXP_DATE, STRIKE, RIGHT)` for exact contract lookups. Bid, ask,
and delta come from the same row; midpoint is `(bid + ask) / 2`.

Requests are passed as bound JSON in temporary CLOBs, including documents larger
than 32 KB. SQLAlchemy manages the connection; its Oracle driver cursor sets
array fetching to 8,192 rows and prefetching to 8,193 rows before execution.
The loader does not inspect catalogs, reflect tables, select sources, or execute
SQL per trade. Schema changes require an explicit update to the static queries.

Feature and price preparation runs once on CPU. Candidate evaluation uses
PyTorch on the selected device, with bounded candidate and observation chunks;
no database queries occur inside evolution. `observation_batch_size` limits
minute workspaces, and `objective_batch_size` limits candidate workspaces.

Optimization and reported performance use the supplied data in sample. Use a
separate date range to assess the resulting rules out of sample.

## Fast custom metrics on CUDA

Reduce along the market-day dimension and score every candidate in the batch
together. Avoid Python loops over candidates or days. Keep arithmetic, temporary
tensors, and results on the input device and dtype; use `returns.new_tensor(...)`
when a tensor constant is needed. Calls such as `.item()`, `float(tensor)`,
`.cpu()`, or `.numpy()` inside the arithmetic path force synchronization or data
transfers and can dominate a small metric's runtime.

Each candidate's result must depend only on its own return series. Avoid
normalizing across candidates, random sampling, mutable counters, input mutation,
or retaining input tensors between calls. Evolution changes candidate ordering
and batch sizes, so these patterns can change the selected strategy as well as
reduce performance.

For a metric with many small tensor operations, compile a pure arithmetic helper
once and reuse it. Keep Python validation and error raising outside that helper.
For example, this metric computes annualized arithmetic return plus 1:

```python
import torch


def _score(returns: torch.Tensor) -> torch.Tensor:
    return (1.0 + 252.0 * returns.mean(dim=1)).clamp_min(1e-6)


# Compilation is lazy; reuse this callable across generations and batches.
_cuda_score = torch.compile(_score, dynamic=True)


def custom_metric(returns: torch.Tensor) -> torch.Tensor:
    if returns.ndim != 2 or min(returns.shape) == 0:
        raise ValueError("Expected a nonempty (candidates, market_days) tensor")
    if not returns.is_floating_point():
        raise TypeError("Returns must be floating point")
    if not bool(torch.isfinite(returns).all()):
        raise ValueError("Returns must be finite")
    return _cuda_score(returns) if returns.is_cuda else _score(returns)
```

The explicit finite check synchronizes CUDA. The public built-in metrics retain
these checks deliberately so invalid input raises a useful Python exception.
Built-ins fuse the finite-value reduction with compiled arithmetic, then inspect
a scalar validity flag outside compilation. Custom metrics can use this pattern
when a separate validation scan is a measurable cost.
When designing a trusted internal path, validate invariant data once before the
repeated arithmetic; retain validation at public boundaries. Keep the helper
graph friendly, with no tensor-dependent Python conditions. The default
`fullgraph=False` permits eager fallback if a compiler cache limit is reached;
`fullgraph=True` can help diagnose graph breaks during development. Replacing
public validation with a device assertion can leave the CUDA context unusable
after invalid input.

Compilation adds a potentially substantial first-call cost and may specialize
on shape, dtype, device, or layout. `dynamic=True` allows varying dimensions, but
guards and layout changes can still trigger another compilation. Warm the shapes
used in the run, cache the compiled callable, and measure compilation separately
from repeated execution. Compiling an already small metric is not always faster;
compare complete objective evaluation as well as the metric itself.

Use a CPU wall-clock timer with `torch.cuda.synchronize()` immediately before
starting and after the measured calls. Warm up first, repeat measurements, and
compare medians for the same inputs and batch sizes. Include validation and any
required transfers when measuring the public metric, since they remain part of
its actual cost.

## Measured CUDA performance

Benchmarks used an RTX 3090, PyTorch 2.15, and CUDA 13.2 in the `torch` conda
environment. The fixed synthetic combined-mode workload has 960 trades, 970
market days, 179 exit observations per trade, 25 features, eight conjunctions,
and candidate batches of 32. Evolution uses 128 candidates for three generations
with progress rendering disabled. Inputs are already on the GPU; timings exclude
Oracle reads, CPU preparation, final allocation, plotting, and CSV export.
These historical measurements preceded final rule simplification; the current
evolution stage also scores its final simplification trials.
These recorded measurements predate threshold granularity and the expanded
feature defaults; the raw records'
source hashes identify the exact implementations measured.

Each retained change was measured before the next change. Warm wall-clock
medians, in milliseconds, progressed as follows:

| Change | Evaluate 32 candidates | Three generations, annualized return |
| --- | ---: | ---: |
| Original implementation; 512-observation chunks | 151.75 | 2,469.33 |
| Compile feature matching | 110.39 | 1,885.89 |
| Increase observation chunks to 32,768 | 5.49 | 146.95 |
| Put reduced dimensions next to each other | 4.73 | 114.18 |
| Final implementation, including metric and validation fusion | 4.87 | 113.99 |

The final baseline recheck measured 174.13 ms for evaluation and 2,641.01 ms for
evolution. Against that recheck, the final implementation is **35.75×** and
**23.17×** faster respectively. Both use nine samples, with five evaluator calls
or two evolution calls per sample. The Windows GPU driver, changing clocks, and
other users of the shared GPU introduce noise; intermediate best timings should
not be interpreted as additional independent gains.

For metrics alone on `(1024, 970)` return tensors, fusing arithmetic and validity
checks reduced annualized return from 0.4125 to 0.3182 ms, Sortino from 1.3953 to
0.2882 ms, and return/ADD from 0.7788 to 0.2669 ms. Public finite-input checks and
Python exceptions remain in place.

The chunk sweep measured 5.49 ms at 32,768 observations, 5.70 ms at 65,536, and
7.10 ms with the whole exit series together. Defaults are therefore 32,768 on
CUDA and 512 on CPU; `observation_batch_size` overrides either. Larger chunks
trade launch overhead for memory: extra peak PyTorch allocations in the final
evaluator recheck were about 18.1 MiB, versus 5.7 MiB for the original chunks.
Deferring history collection and caching observation indices provided no consistent improvement
and were reverted. The index-cache trial measured 5.42 versus 5.20 ms for
evaluation and 132.86 versus 127.81 ms for evolution; the history trial measured
174.47 versus 158.96 ms for its evolution control.

Compiled predicates and metrics add startup cost. The final first invocation took
3.084 seconds for evaluation and 3.623 seconds for evolution. An existing on-disk
compiler cache may be reused, so these are first-call measurements rather than
strict cold-cache measurements. Shape variants can compile again; reaching a
compiler cache limit permits eager fallback. Warm timings synchronize CUDA
before and after every measured sample. The
[raw records](../../benchmarks/results/strategy_optimizer_rtx3090.json) retain
configuration, source hashes, seeds, every timing sample, result summaries, and
peak allocated memory, including reverted experiments.

Run from the repository root in the `torch` environment:

```bash
conda activate torch
python -m benchmarks.benchmark_strategy_optimization \
  --stage evaluator --label local-evaluator --device cuda \
  --observation-batch-size 32768 --repeats 9 --iterations 5 \
  --output benchmark-results/strategy-optimization.json
python -m benchmarks.benchmark_strategy_optimization \
  --stage evolution --label local-evolution --device cuda \
  --observation-batch-size 32768 --repeats 9 --iterations 2 \
  --output benchmark-results/strategy-optimization.json
python -m benchmarks.benchmark_strategy_optimization \
  --stage metrics --metric return_over_add --label local-metric --device cuda \
  --iterations 20 --output benchmark-results/strategy-optimization.json
```

Use `--profile /tmp/strategy-optimization-trace.json` to export a profiler trace
after timing; profiling itself is excluded from the samples.
