"""Polars contract tests for OptionAlpha table handling."""

from datetime import date, time

import polars as pl
import torch

from ifera.optionalpha import parse_trade_log, prepare_splits


def test_parse_trade_log_returns_polars_dataframe():
    """Trade log parsing should return a Polars DataFrame with a date column."""
    html = """
    <grid>
        <row>
            <bd>
                <div class="cell symbol">
                    <div class="clip">
                        <span class="sym">SPX</span>
                        <span>Long Call</span>
                    </div>
                </div>
                <div class="cell closeTime">
                    <div class="clip">Jan 10, 2022</div>
                    <div class="clip">3:47pm → 4:00pm</div>
                </div>
                <div class="cell status">
                    <span class="lbl">Expired</span>
                </div>
                <div class="cell risk">
                    <span class="val pos">$380</span>
                </div>
                <div class="cell pnl">
                    <span class="val pos pnl">$1,149</span>
                </div>
            </bd>
        </row>
    </grid>
    """

    df = parse_trade_log(html)

    assert isinstance(df, pl.DataFrame)
    assert "date" in df.columns
    assert df.select("date").to_series().to_list() == [date(2022, 1, 10)]


def test_prepare_splits_accepts_polars_tables():
    """Split preparation should accept Polars tables as input."""
    trades_df = pl.DataFrame(
        {
            "date": [date(2022, 1, 10), date(2022, 1, 11)],
            "risk": [100.0, 120.0],
            "profit": [50.0, -20.0],
            "start_time": [time(9, 30), time(10, 0)],
        }
    )
    filters_df = pl.DataFrame(
        {
            "date": [date(2022, 1, 10), date(2022, 1, 11)],
            "filter_a": [1.0, 2.0],
        }
    )

    x_tensor, y_tensor, splits = prepare_splits(
        trades_df=trades_df,
        filters_df=filters_df,
        spread_width=20,
        left_only_filters=[],
        right_only_filters=[],
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_depth=1,
    )

    assert x_tensor.shape[0] == 2
    assert y_tensor.shape[0] == 2
    assert len(splits) > 0
