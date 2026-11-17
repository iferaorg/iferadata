"""Offline tests for published event calendars, complete caches, and offsets."""

import calendar
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from ifera.strategy_optimization import _event_sources as sources
from ifera.strategy_optimization._event_sources import (
    EventCalendarError,
    EventYear,
    OfficialEventProvider,
    canonical_event_type,
)
from ifera.strategy_optimization._events import (
    EventCalendar,
    event_flags,
    load_event_dates,
)


@pytest.fixture(autouse=True)
def _disable_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise requests.ConnectionError("Network disabled in event unit tests")

    monkeypatch.setattr(sources.requests, "get", blocked)


def _year(event="CPI Release", year=2024, dates=None):
    return EventYear(
        event,
        year,
        (date(year, 1, 10),) if dates is None else dates,
        ("https://example.test/verified-complete-year",),
        True,
    )


def _provider(documents):
    provider = OfficialEventProvider()
    provider._documents.update(
        {key: value.encode() for key, value in documents.items()}
    )
    return provider


@pytest.mark.parametrize("name", ["CPI Release", "cpi_release", " CPI RELEASE "])
def test_event_names_are_canonicalized(name):
    assert canonical_event_type(name) == "CPI Release"


@pytest.mark.parametrize("name", ["unknown", "", None, 1])
def test_unknown_event_names_are_rejected(name):
    with pytest.raises(ValueError, match="Unknown event type"):
        canonical_event_type(name)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"complete": False}, "Incomplete calendar"),
        ({"year": True}, "integer"),
        ({"year": 1800}, "between"),
        ({"sources": ()}, "provenance"),
        ({"dates": (date(2023, 1, 1),)}, "belong to"),
        ({"dates": (datetime(2024, 1, 1),)}, "belong to"),
        ({"dates": (date(2024, 2, 1), date(2024, 1, 1))}, "sorted and unique"),
        ({"dates": (date(2024, 1, 1),) * 2}, "sorted and unique"),
    ],
)
def test_provider_year_requires_explicit_complete_valid_dates(overrides, match):
    fields = {
        "event_type": "CPI Release",
        "year": 2024,
        "dates": (),
        "sources": ("publisher",),
        "complete": True,
        **overrides,
    }
    with pytest.raises(EventCalendarError, match=match):
        EventYear(**fields)


def test_signed_offsets_use_nyse_sessions_and_preserve_order_and_duplicates():
    dates = [date(2023, 12, 29), date(2024, 1, 2), date(2023, 12, 29)]
    events = {"CPI Release": (date(2024, 1, 2),)}
    np.testing.assert_equal(
        event_flags(dates, "CPI Release", 1, events=events), [1, 0, 1]
    )
    np.testing.assert_equal(
        event_flags(dates, "CPI Release", 0, events=events), [0, 1, 0]
    )
    np.testing.assert_equal(
        event_flags([date(2024, 1, 3)], "CPI Release", -1, events=events), [1]
    )


def test_economic_releases_on_market_holidays_are_not_rolled():
    events = {"Nonfarm Payrolls": (date(2025, 4, 18),)}
    dates = [date(2025, 4, 17), date(2025, 4, 18), date(2025, 4, 21)]
    for offset in (-1, 0, 1):
        assert not event_flags(dates, "Nonfarm Payrolls", offset, events=events).any()


def test_aware_timestamps_use_new_york_date():
    result = event_flags(
        [datetime(2024, 1, 3, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, 10)],
        "CPI Release",
        0,
        events={"CPI Release": (date(2024, 1, 2),)},
    )
    np.testing.assert_equal(result, [1, 1])
    assert result.dtype == np.float64


@pytest.mark.parametrize("days", [True, 1.5, -11, 11])
def test_invalid_offsets_are_rejected(days):
    with pytest.raises(ValueError, match="between -10 and 10"):
        event_flags([date(2024, 1, 2)], "CPI Release", days, events={})


def test_invalid_observations_and_preloaded_dates_are_rejected():
    with pytest.raises(ValueError, match="dates or datetimes"):
        event_flags(["2024-01-02"], "CPI Release", 0, events={})
    with pytest.raises(EventCalendarError, match="do not include"):
        event_flags([date(2024, 1, 2)], "CPI Release", 0, events={})
    with pytest.raises(ValueError, match="date objects"):
        event_flags(
            [date(2024, 1, 2)],
            "CPI Release",
            0,
            events={"CPI Release": (datetime(2024, 1, 2),)},
        )
    assert event_flags([], "CPI Release", 0, events={}).shape == (0,)


def test_preload_groups_events_and_boundary_years_once(tmp_path):
    calls = []

    def provider(event, year):
        calls.append((event, year))
        return _year(event, year)

    source = EventCalendar(tmp_path, provider)
    dates = [date(2024, 1, 2), date(2024, 12, 31)]
    result = load_event_dates(dates, ["cpi_release", "CPI Release"], calendar=source)
    assert calls == [("CPI Release", year) for year in (2023, 2024, 2025)]
    assert len(result["CPI Release"]) == 3
    load_event_dates(dates, ["CPI Release"], calendar=source)
    assert len(calls) == 3


def test_preload_only_selected_offsets_avoids_unpublished_adjacent_year(tmp_path):
    calls = []

    def provider(event, year):
        calls.append(year)
        return _year(event, year)

    result = load_event_dates(
        [date(2024, 12, 31)],
        ["CPI Release"],
        calendar=EventCalendar(tmp_path, provider),
        offsets=[0],
    )
    assert calls == [2024]
    assert result["CPI Release"] == (date(2024, 1, 10),)


@pytest.mark.parametrize("offsets", [[], [True], [11], [0.5]])
def test_invalid_preload_offsets_rejected(offsets):
    with pytest.raises(ValueError, match="Event offsets"):
        load_event_dates([], ["CPI Release"], offsets=offsets)


def test_complete_cache_round_trip_empty_year_and_provenance(tmp_path):
    source = EventCalendar(tmp_path, lambda *_: pytest.fail("unexpected network"))
    source.store(_year())
    source.store(_year("PPI Release", dates=()))
    paths = sorted(tmp_path.rglob("*.csv"))
    assert len(paths) == 2
    metadata = json.loads(paths[0].read_text().splitlines()[0][2:])
    assert metadata["coverage_start"] == "2024-01-01"
    assert metadata["coverage_end"] == "2024-12-31"
    assert metadata["sources"] == ["https://example.test/verified-complete-year"]
    other = EventCalendar(tmp_path, lambda *_: pytest.fail("unexpected network"))
    assert other.dates("CPI Release", 2024) == (date(2024, 1, 10),)
    assert other.dates("PPI Release", 2024) == ()
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize(
    "damage", ["header", "count", "coverage", "date", "event", "complete"]
)
def test_invalid_or_partial_caches_raise_instead_of_silently_false(tmp_path, damage):
    source = EventCalendar(tmp_path)
    source.store(_year())
    path = next(tmp_path.rglob("*.csv"))
    lines = path.read_text().splitlines()
    metadata = json.loads(lines[0][2:])
    if damage == "header":
        lines = lines[1:]
    elif damage == "date":
        lines[-1] = "2023-01-10"
    else:
        key, value = {
            "count": ("count", 2),
            "coverage": ("coverage_end", "2024-06-30"),
            "event": ("event_type", "PPI Release"),
            "complete": ("complete", False),
        }[damage]
        metadata[key] = value
        lines[0] = "# " + json.dumps(metadata)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(EventCalendarError, match="Invalid event cache"):
        EventCalendar(tmp_path).dates("CPI Release", 2024)


def test_expired_cache_refreshes_once_per_instance(tmp_path):
    source = EventCalendar(tmp_path)
    source.store(_year())
    path = next(tmp_path.rglob("*.csv"))
    lines = path.read_text().splitlines()
    metadata = json.loads(lines[0][2:])
    metadata["fetched_at"] = (
        datetime.now(timezone.utc) - timedelta(days=366)
    ).isoformat()
    lines[0] = "# " + json.dumps(metadata)
    path.write_text("\n".join(lines) + "\n")
    calls = []

    def provider(event, year):
        calls.append(year)
        return _year(event, year, (date(year, 2, 10),))

    refreshed = EventCalendar(tmp_path, provider)
    assert refreshed.dates("CPI Release", 2024) == (date(2024, 2, 10),)
    refreshed.dates("CPI Release", 2024)
    assert calls == [2024]


def test_unavailable_year_includes_event_year_and_cache_path(tmp_path):
    def provider(*_):
        raise EventCalendarError("publisher blocked the request")

    with pytest.raises(EventCalendarError, match=r"CPI Release, 2024.*2024.csv"):
        EventCalendar(tmp_path, provider).dates("CPI Release", 2024)
    assert not list(tmp_path.rglob("*.csv"))


def test_exchange_rules_include_holidays_month_end_quarter_and_year_roll():
    provider = OfficialEventProvider()
    assert date(2025, 4, 17) in provider("Monthly Expiration", 2025).dates
    assert date(2025, 7, 3) in provider("First Weekly", 2025).dates
    assert date(2020, 12, 31) in provider("First Weekly", 2020).dates
    assert all(item.year == 2021 for item in provider("First Weekly", 2021).dates)
    assert provider("Triple Witching", 2025).dates == (
        date(2025, 3, 21),
        date(2025, 6, 20),
        date(2025, 9, 19),
        date(2025, 12, 19),
    )
    assert date(2025, 8, 29) in provider("EndofMonth", 2025).dates
    assert provider("EndofQuarter", 2025).dates == (
        date(2025, 3, 31),
        date(2025, 6, 30),
        date(2025, 9, 30),
        date(2025, 12, 31),
    )


def test_downloads_are_shared_and_http_errors_are_explicit(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        response = requests.Response()
        response.status_code = 200
        response._content = b"calendar"
        return response

    monkeypatch.setattr(sources.requests, "get", get)
    provider = OfficialEventProvider()
    assert provider._download("https://example.test/calendar") == b"calendar"
    provider._download("https://example.test/calendar")
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == (10, 30)
    monkeypatch.setattr(
        sources.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.Timeout()),
    )
    with pytest.raises(EventCalendarError, match="Cannot download event calendar"):
        provider._download("https://example.test/blocked")


def test_public_calendar_download_preserves_working_requests_user_agent(monkeypatch):
    def get(_url, **kwargs):
        if "User-Agent" in kwargs.get("headers", {}):
            raise requests.ReadTimeout(
                "Publisher tarpits the custom application user agent"
            )
        response = requests.Response()
        response.status_code = 200
        response._content = b"published calendar"
        return response

    monkeypatch.setattr(sources.requests, "get", get)
    assert (
        OfficialEventProvider()._download("https://fred.stlouisfed.org/calendar")
        == b"published calendar"
    )


def test_fomc_current_calendar_uses_only_scheduled_final_meeting_days():
    months = (
        "January",
        "March",
        "April",
        "June",
        "July",
        "September",
        "October",
        "December",
    )
    rows = "".join(
        f'<div class="fomc-meeting"><div class="fomc-meeting__month">{month}</div><div class="fomc-meeting__date">15-16*</div></div>'
        for month in months
    )
    rows += '<div class="fomc-meeting"><div class="fomc-meeting__month">August</div><div class="fomc-meeting__date">22 (notation vote)</div></div>'
    provider = _provider(
        {sources._FED: f'<div class="panel"><h4>2024 FOMC Meetings</h4>{rows}</div>'}
    )
    result = provider("FOMC Meeting", 2024)
    assert len(result.dates) == 8
    assert result.dates[0] == date(2024, 1, 16)
    assert date(2024, 8, 22) not in result.dates


def test_bls_reads_actual_revised_dates_and_requires_complete_calendar_rows():
    headings = "".join(f"<h2>{month} 2024</h2>" for month in calendar.month_name[1:])
    rows = "".join(
        f"<tr><td>Monday, {month} 15, 2024</td><td>08:30 AM</td><td>Consumer Price Index for prior month</td></tr>"
        for month in calendar.month_name[1:]
    )
    url = "https://www.bls.gov/schedule/2024/home.htm"
    provider = _provider({url: headings + "<table>" + rows + "</table>"})
    assert provider._bls_published("CPI Release", 2024).dates[1] == date(2024, 2, 15)
    provider._documents[url] = (
        headings + "<table>" + rows.split("</tr>")[0] + "</tr></table>"
    ).encode()
    with pytest.raises(EventCalendarError, match="rows do not cover"):
        provider._bls_published("CPI Release", 2024)


def test_alfred_historical_fallback_checks_release_identity_and_year_coverage():
    url = "https://alfred.stlouisfed.org/release/downloaddates?ff=txt&rid=10"
    provider = _provider(
        {
            url: "Release: Consumer Price Index\nFile Created: 2025-01-02\nRelease Dates:\n2023-12-10\n2024-01-11\n2024-12-11\n"
        }
    )
    result = provider._alfred("CPI Release", 2024)
    assert result.dates == (date(2024, 1, 11), date(2024, 12, 11))
    assert result.sources == (url,)
    with pytest.raises(EventCalendarError, match="fully cover"):
        provider._alfred("CPI Release", 2025)


def _fred_html(year=2026):
    fields = {
        "vs": f"{year}-01-01",
        "ve": f"{year}-12-31",
        "min-rd": "2025-01-01",
        "max-rd": "2027-12-08",
    }
    html = "".join(
        f'<input id="rc-{key}" value="{value}">' for key, value in fields.items()
    )
    rows = "".join(
        f"<tr><td><span>Tuesday {month} 10, {year}</span></td></tr>"
        '<tr><td><a href="/release?rid=10">Consumer Price Index</a></td></tr>'
        for month in calendar.month_name[1:]
    )
    return (
        html
        + '<div id="release-dates-pager"><table>'
        + rows
        + "</table>Releases 1 - 12 of 12</div>"
    )


def test_fred_calendar_includes_published_future_dates_and_validates_filters():
    url = "https://fred.stlouisfed.org/releases/calendar?rid=10&vs=2026-01-01&ve=2026-12-31"
    provider = _provider({url: _fred_html()})
    result = provider._fred_calendar("CPI Release", 2026)
    assert result.dates[-1] == date(2026, 12, 10)
    assert result.sources == (url,)
    provider._documents[url] = (
        _fred_html().replace('value="2026-12-31"', 'value="2026-01-31"').encode()
    )
    with pytest.raises(EventCalendarError, match="does not fully cover"):
        provider._fred_calendar("CPI Release", 2026)


def test_fred_calendar_rejects_partial_pagination_and_unsupported_year():
    url = "https://fred.stlouisfed.org/releases/calendar?rid=10&vs=2026-01-01&ve=2026-12-31"
    provider = _provider(
        {url: _fred_html().replace("Releases 1 - 12 of 12", "Releases 1 - 10 of 12")}
    )
    with pytest.raises(EventCalendarError, match="incomplete or paginated"):
        provider._fred_calendar("CPI Release", 2026)
    older = "https://fred.stlouisfed.org/releases/calendar?rid=10&vs=2024-01-01&ve=2024-12-31"
    provider._documents[older] = _fred_html(2024).encode()
    with pytest.raises(EventCalendarError, match="does not fully cover"):
        provider._fred_calendar("CPI Release", 2024)


def test_bea_archive_uses_filter_option_and_actual_publication_dates():
    archive = '<select name="created_1"><option value="7">2024</option></select>'
    url = (
        sources._BEA_ARCHIVE
        + "?field_related_product_target_id=All&created_1=7&title=Personal+Income+and+Outlays"
    )
    results = '<table><tr><td>Personal Income and Outlays, January 2024</td><td><time datetime="2024-02-29T08:30:00-05:00">February 29,2024</time></td></tr></table>'
    provider = _provider({sources._BEA_ARCHIVE: archive, url: results})
    assert provider("PCE Release", 2024).dates == (date(2024, 2, 29),)
    provider._documents[url] = results.replace("2024-02-29T", "2023-02-28T").encode()
    with pytest.raises(EventCalendarError, match="ignored its year filter"):
        provider._bea_archive(2024)


def test_current_bea_year_uses_full_schedule_instead_of_partial_archive():
    year = date.today().year
    archive = f'<select name="created_1"><option value="2">{year}</option></select>'
    filtered = (
        sources._BEA_ARCHIVE
        + "?field_related_product_target_id=All&created_1=2&title=Personal+Income+and+Outlays"
    )
    published = (
        "<table>"
        + "".join(
            f'<tr><td>Personal Income and Outlays, prior month</td><td><time datetime="{year}-{month:02}-15T08:30:00-05:00">published</time></td></tr>'
            for month in range(1, 12)
        )
        + "</table>"
    )
    scheduled = (
        '<table id="release-schedule-table">'
        + "".join(
            f'<tr><td class="release-date">{name} 15</td><td class="release-title">Personal Income and Outlays, prior month</td><td><a href="/news/{year}/release">View</a></td></tr>'
            for name in calendar.month_name[1:]
        )
        + "</table>"
    )
    provider = _provider(
        {
            sources._BEA_ARCHIVE: archive,
            filtered: published,
            sources._BEA_SCHEDULE: scheduled,
        }
    )
    result = provider("PCE Release", year)
    assert result.dates[-1] == date(year, 12, 15)
    assert result.sources == (sources._BEA_SCHEDULE,)


def test_completed_bea_year_missing_archive_uses_actual_fred_dates():
    year = date.today().year - 1
    url = f"https://fred.stlouisfed.org/releases/calendar?rid=54&vs={year}-01-01&ve={year}-12-31"
    recorded = (
        _fred_html(year)
        .replace("rid=10", "rid=54")
        .replace("Consumer Price Index", "Personal Income and Outlays")
    )
    stale = (
        '<table id="release-schedule-table">'
        + "".join(
            f'<tr><td class="release-date">{month} 15</td><td class="release-title">Personal Income and Outlays, prior month</td><td><a href="/news/{year}/release">View</a></td></tr>'
            for month in calendar.month_name[1:]
        )
        + "</table>"
    )
    provider = _provider(
        {
            sources._BEA_ARCHIVE: "<select></select>",
            sources._BEA_SCHEDULE: stale,
            url: recorded,
        }
    )
    result = provider("PCE Release", year)
    assert result.dates[-1] == date(year, 12, 10)
    assert result.sources == (url,)


def test_ism_published_table_keeps_exceptional_dates_and_rejects_other_years():
    rows = "".join(
        f"<tr><td>{month} 2026</td><td>{5 if index == 1 else 1}</td><td>7<sup>**</sup></td></tr>"
        for index, month in enumerate(calendar.month_name[1:], 1)
    )
    provider = _provider({sources._ISM: "<table>" + rows + "</table>"})
    assert provider._ism_published("Manufacturing PMI", 2026).dates[0] == date(
        2026, 1, 5
    )
    assert provider._ism_published("Services PMI", 2026).dates[0] == date(2026, 1, 7)
    with pytest.raises(EventCalendarError, match="complete accessible"):
        provider._ism_published("Services PMI", 2025)


def test_ism_recorded_history_requires_twelve_months_from_correct_series():
    rows = [
        {"indicator": "us-ism-services", "at": f"2024-{month:02}-03T15:00:00.000Z"}
        for month in range(1, 13)
    ]
    payload = "6:" + json.dumps({"history": rows}) + "\n"
    script = "<script>self.__next_f.push([1," + json.dumps(payload) + "])</script>"
    url = "https://makkler.com/labs/economic-calendar/us-ism-services"
    provider = _provider({url: script})
    assert len(provider._ism_history("Services PMI", 2024).dates) == 12
    with pytest.raises(EventCalendarError, match="incomplete"):
        provider._ism_history("Services PMI", 2025)


def _press_card(published, slug):
    return (
        '<a class="newsreleaseconsolidatelink" href="/news-releases/' + slug + '">'
        "<h3><small>" + published.strftime("%b %d, %Y") + ", 10:00 ET</small>"
        "Monthly ISM report</h3></a>"
    )


def test_ism_press_archive_uses_publication_year_and_legacy_services_name():
    first = sources._ISM_PRESS + "?page=1&pagesize=100"
    second = sources._ISM_PRESS + "?page=2&pagesize=100"
    page1 = _press_card(
        date(2025, 1, 7),
        "services-pmi-at-54-december-2024-ism-services-pmi-report-1.html",
    )
    page1 += _press_card(
        date(2024, 12, 4),
        "nmi-at-54-november-non-manufacturing-ism-report-on-business-1.html",
    )
    page2 = "".join(
        _press_card(
            date(2024, month, 3),
            "nmi-at-54-prior-month-non-manufacturing-ism-report-on-business-1.html",
        )
        for month in range(1, 12)
    )
    page2 += _press_card(
        date(2023, 12, 5),
        "nmi-at-54-november-non-manufacturing-ism-report-on-business-2.html",
    )
    provider = _provider({first: page1, second: page2})
    result = provider._ism_press_history("Services PMI", 2024)
    assert len(result.dates) == 12
    assert result.dates[-1] == date(2024, 12, 4)
    assert result.sources == (first, second)
    with pytest.raises(EventCalendarError, match="incomplete"):
        provider._ism_press_history("Manufacturing PMI", 2024)


def test_ism_press_archive_accepts_explicit_next_release_announcement():
    year = date.today().year
    source = sources._ISM_PRESS + "?page=1&pagesize=100"
    slug = "manufacturing-pmi-at-54-current-ism-manufacturing-pmi-report-1.html"
    cards = "".join(_press_card(date(year, month, 2), slug) for month in range(1, 12))
    cards += _press_card(date(year - 1, 12, 1), slug)
    report = "https://www.prnewswire.com/news-releases/" + slug
    provider = _provider(
        {
            source: cards,
            report: f"The next ISM Manufacturing PMI Report will be released at 10:00 a.m. ET on Tuesday, December 1, {year}.",
        }
    )
    result = provider._ism_press_history("Manufacturing PMI", year)
    assert len(result.dates) == 12
    assert result.dates[-1] == date(year, 12, 1)
    assert result.sources[-1] == report


def test_ism_press_archive_recognizes_august_2020_services_rename_release():
    url = sources._ISM_PRESS + "?page=3&pagesize=100"
    slug = "services-pmi-formerly-non-manufacturing-nmi-at-58-1-301106674.html"
    provider = _provider({url: _press_card(date(2020, 8, 5), slug)})
    reports, _ = provider._ism_press_page(url, "Services PMI")
    assert reports == [
        (date(2020, 8, 5), "https://www.prnewswire.com/news-releases/" + slug)
    ]


def test_sentiment_pdf_contains_preliminary_and_final_with_year_boundary(monkeypatch):
    url = "https://data.sca.isr.umich.edu/fetchdoc.php?docid=release"
    document = "Release Dates\n1991 through 2027\nDec 2023 12/15/2023 1/5/2024\n"
    document += "\n".join(
        f"Jan 2024 {month}/12/2024 {month}/26/2024" for month in range(1, 13)
    )
    monkeypatch.setattr(
        sources,
        "PdfReader",
        lambda _stream: SimpleNamespace(
            pages=[SimpleNamespace(extract_text=lambda: document)]
        ),
    )
    provider = _provider(
        {
            sources._MICHIGAN: '<a href="fetchdoc.php?docid=release">Historical Prelim/Final Release Dates</a>',
            url: "PDF",
        }
    )
    result = provider("Consumer Sentiment", 2024)
    assert len(result.dates) == 25
    assert date(2024, 1, 5) in result.dates
    assert date(2024, 1, 12) in result.dates
    assert date(2024, 1, 26) in result.dates
    with pytest.raises(EventCalendarError, match="fully cover"):
        provider("Consumer Sentiment", 1991)
