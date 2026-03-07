import datetime as dt

import pytest

from ifera.time_utils import derive_end_time_and_steps, parse_date, parse_timedelta


def test_parse_timedelta_supports_interval_and_clock_formats() -> None:
    assert parse_timedelta("30m") == dt.timedelta(minutes=30)
    assert parse_timedelta("1h") == dt.timedelta(hours=1)
    assert parse_timedelta("17:00:00") == dt.timedelta(hours=17)
    assert parse_timedelta("-06:00:00") == dt.timedelta(hours=-6)


def test_parse_timedelta_rejects_invalid_text() -> None:
    with pytest.raises(ValueError, match="Cannot parse timedelta"):
        parse_timedelta("not-a-duration")


def test_parse_date_from_iso_text() -> None:
    assert parse_date("2020-01-01") == dt.date(2020, 1, 1)


def test_derive_end_time_and_steps_matches_config_behavior() -> None:
    end_time, total_steps = derive_end_time_and_steps(
        time_step=dt.timedelta(minutes=30),
        trading_start=dt.timedelta(hours=-6),
        trading_end=dt.timedelta(hours=17),
    )

    assert end_time == dt.timedelta(hours=22, minutes=30)
    assert total_steps == 46


def test_derive_end_time_and_steps_with_large_step() -> None:
    end_time, total_steps = derive_end_time_and_steps(
        time_step=dt.timedelta(hours=30),
        trading_start=dt.timedelta(hours=-6),
        trading_end=dt.timedelta(hours=17),
    )

    assert end_time == dt.timedelta(0)
    assert total_steps == 1
