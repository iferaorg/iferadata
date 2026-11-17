"""Yearly event caches and vectorized flags at signed NYSE-session offsets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from ifera.allocation.portfolio_allocation import _nyse_market_dates
from ifera.settings import settings

from ._event_sources import (
    EVENT_TYPES,
    EventCalendarError,
    EventYear,
    OfficialEventProvider,
    canonical_event_type,
)

_NEW_YORK = ZoneInfo("America/New_York")
_CACHE_VERSION = 1
_MAX_OFFSET = 10
__all__ = ["EVENT_TYPES", "EventCalendar", "event_flags", "load_event_dates"]


def _observation_dates(timestamps: Sequence[date | datetime]) -> tuple[date, ...]:
    """Use New York dates for aware timestamps and preserve naive market time."""
    result = []
    for timestamp in timestamps:
        if isinstance(timestamp, datetime):
            result.append(
                timestamp.astimezone(_NEW_YORK).date()
                if timestamp.tzinfo is not None
                else timestamp.date()
            )
        elif isinstance(timestamp, date):
            result.append(timestamp)
        else:
            raise ValueError("Event observations must be dates or datetimes")
    return tuple(result)


def _sessions(dates: Sequence[date]) -> NDArray[np.datetime64]:
    """Pad observations enough to count ten sessions across historical closures."""
    return np.asarray(
        _nyse_market_dates(
            min(dates) - timedelta(days=60), max(dates) + timedelta(days=60)
        )["date"].to_list(),
        dtype="datetime64[D]",
    )


def _source_years(dates: Sequence[date], offsets: Sequence[int]) -> range:
    """Include event years reached by the selected signed session offsets."""
    sessions = _sessions(dates)
    first = int(np.searchsorted(sessions, np.datetime64(min(dates))))
    last = int(np.searchsorted(sessions, np.datetime64(max(dates)), side="right")) - 1
    start = sessions[first + min(offsets)].astype(object)
    end = sessions[last + max(offsets)].astype(object)
    return range(start.year, end.year + 1)


class EventCalendar:
    """Load complete event years once, persisting provenance in atomic CSV files.

    A cache file starts with a JSON metadata comment followed by a single
    ``date`` CSV column. Explicit complete-year metadata supports genuinely empty
    years without confusing them with incomplete downloads. Files refresh after
    365 days; an instance reuses each loaded year throughout preparation.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        provider: Callable[[str, int], EventYear] | None = None,
    ) -> None:
        self.cache_dir = (
            Path(settings.DATA_FOLDER) / "cache" / "strategy_optimization" / "events"
            if cache_dir is None
            else Path(cache_dir)
        )
        self._provider = OfficialEventProvider() if provider is None else provider
        self._loaded: dict[tuple[str, int], tuple[date, ...]] = {}

    def _path(self, event_type: str, year: int) -> Path:
        """Use canonical names to prevent aliases from creating duplicate caches."""
        slug = re.sub(r"[^a-z0-9]+", "_", event_type.casefold()).strip("_")
        return self.cache_dir / slug / f"{year}.csv"

    def _read(self, path: Path, event_type: str, year: int) -> EventYear | None:
        """Reject malformed or partial cache files and refresh only expired ones."""
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                header = stream.readline()
                if not header.startswith("# "):
                    raise ValueError("missing complete-year metadata")
                metadata = json.loads(header[2:])
                if not isinstance(metadata, dict) or any(
                    metadata.get(key) != value
                    for key, value in {
                        "version": _CACHE_VERSION,
                        "event_type": event_type,
                        "year": year,
                        "coverage_start": date(year, 1, 1).isoformat(),
                        "coverage_end": date(year, 12, 31).isoformat(),
                    }.items()
                ):
                    raise ValueError("incorrect event, year, or coverage metadata")
                reader = csv.DictReader(stream)
                if reader.fieldnames != ["date"]:
                    raise ValueError("expected one date column")
                dates = tuple(date.fromisoformat(row["date"]) for row in reader)
                if metadata.get("count") != len(dates):
                    raise ValueError("date count does not match complete-year metadata")
                sources = metadata.get("sources")
                if not isinstance(sources, list):
                    raise ValueError("missing source provenance")
                result = EventYear(
                    event_type,
                    year,
                    dates,
                    tuple(sources),
                    metadata.get("complete") is True,
                )
                fetched = datetime.fromisoformat(metadata["fetched_at"])
                if fetched.tzinfo is None:
                    raise ValueError("fetched_at must include a timezone")
                age = datetime.now(timezone.utc) - fetched
                if age > timedelta(days=365):
                    return None
                if age < timedelta(days=-1):
                    raise ValueError("fetched_at is in the future")
                return result
        except (KeyError, TypeError, ValueError, csv.Error) as exc:
            raise EventCalendarError(f"Invalid event cache {path}: {exc}") from exc

    def store(self, event_year: EventYear) -> None:
        """Save a verified or manually supplied complete year with its provenance."""
        path = self._path(event_year.event_type, event_year.year)
        metadata: dict[str, Any] = {
            "version": _CACHE_VERSION,
            "event_type": event_year.event_type,
            "year": event_year.year,
            "coverage_start": date(event_year.year, 1, 1).isoformat(),
            "coverage_end": date(event_year.year, 12, 31).isoformat(),
            "complete": event_year.complete,
            "sources": list(event_year.sources),
            "count": len(event_year.dates),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f"{event_year.year}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write("# " + json.dumps(metadata, sort_keys=True) + "\n")
                writer = csv.writer(stream)
                writer.writerow(["date"])
                writer.writerows((item.isoformat(),) for item in event_year.dates)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._loaded[(event_year.event_type, event_year.year)] = event_year.dates

    def dates(self, event_type: str, year: int) -> tuple[date, ...]:
        """Return complete-year dates, raising when coverage cannot be established."""
        event_type = canonical_event_type(event_type)
        if (
            isinstance(year, bool)
            or not isinstance(year, int)
            or not 1900 <= year <= 9998
        ):
            raise ValueError("Event year must be an integer between 1900 and 9998")
        key = event_type, year
        if key in self._loaded:
            return self._loaded[key]
        path = self._path(*key)
        cached = self._read(path, *key)
        if cached is None:
            try:
                loaded = self._provider(*key)
            except EventCalendarError as exc:
                raise EventCalendarError(
                    f"Missing complete event data for {event_type}, {year}: {exc}. "
                    f"Supply a verified complete year with EventCalendar.store(); cache: {path}"
                ) from exc
            if (
                not isinstance(loaded, EventYear)
                or (loaded.event_type, loaded.year) != key
            ):
                raise EventCalendarError(
                    f"Provider returned the wrong event calendar for {key}"
                )
            self.store(loaded)
        else:
            self._loaded[key] = cached.dates
        return self._loaded[key]


def load_event_dates(
    timestamps: Sequence[date | datetime],
    event_types: Sequence[str],
    *,
    calendar: EventCalendar | None = None,
    offsets: Sequence[int] | None = None,
) -> dict[str, tuple[date, ...]]:
    """Preload each event/year once, covering selected session offsets.

    With no offsets specified, include the full allowed ten-session lookaround.
    Passing selected offsets avoids requiring irrelevant unpublished future years.
    """
    dates = _observation_dates(timestamps)
    kinds = tuple(dict.fromkeys(canonical_event_type(value) for value in event_types))
    requested = (-_MAX_OFFSET, _MAX_OFFSET) if offsets is None else tuple(offsets)
    if not requested or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -_MAX_OFFSET <= value <= _MAX_OFFSET
        for value in requested
    ):
        raise ValueError("Event offsets must be nonempty integers between -10 and 10")
    if not dates:
        return {kind: () for kind in kinds}
    source = EventCalendar() if calendar is None else calendar
    years = _source_years(dates, requested)
    return {
        kind: tuple(item for year in years for item in source.dates(kind, year))
        for kind in kinds
    }


def event_flags(
    timestamps: Sequence[date | datetime],
    event_type: str,
    days: int,
    *,
    events: Mapping[str, Sequence[date]] | None = None,
) -> NDArray[np.float64]:
    """Test whether observation + days NYSE sessions is a published event date.

    Positive offsets look ahead; negative offsets look behind. Actual economic
    releases on closed days are not rolled onto a trading day. A preloaded event
    mapping makes this operation entirely offline; otherwise required years are
    loaded before evaluating any observations.
    """
    event_type = canonical_event_type(event_type)
    if isinstance(days, bool) or not isinstance(days, int) or not -10 <= days <= 10:
        raise ValueError("Event days must be an integer between -10 and 10")
    dates = _observation_dates(timestamps)
    if not dates:
        return np.empty(0, dtype=np.float64)
    available = (
        load_event_dates(dates, [event_type], offsets=[days])
        if events is None
        else events
    )
    if event_type not in available:
        raise EventCalendarError(f"Preloaded events do not include {event_type}")
    published = available[event_type]
    if any(
        not isinstance(item, date) or isinstance(item, datetime) for item in published
    ):
        raise ValueError("Preloaded event dates must be date objects")
    observations = np.asarray(dates, dtype="datetime64[D]")
    sessions = _sessions(dates)
    indices = np.searchsorted(sessions, observations)
    valid = sessions[indices] == observations
    targets = np.full(
        observations.shape, np.datetime64("NaT", "D"), dtype="datetime64[D]"
    )
    targets[valid] = sessions[indices[valid] + days]
    return np.isin(targets, np.asarray(published, dtype="datetime64[D]")).astype(
        np.float64
    )
