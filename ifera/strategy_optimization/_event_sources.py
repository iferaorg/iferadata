"""Complete published economic calendars and deterministic exchange-date rules."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import BytesIO
import json
import re
from urllib.parse import urlencode, urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from bs4.element import Tag
from pypdf import PdfReader
import requests

from ifera.allocation.portfolio_allocation import _nyse_market_dates

EVENT_TYPES = (
    "FOMC Meeting",
    "CPI Release",
    "PPI Release",
    "PCE Release",
    "Nonfarm Payrolls",
    "Triple Witching",
    "Monthly Expiration",
    "EndofMonth",
    "EndofQuarter",
    "First Weekly",
    "Manufacturing PMI",
    "Services PMI",
    "Consumer Sentiment",
)
_RULE_EVENTS = frozenset(EVENT_TYPES[5:10])
_MONTHS = {
    name.casefold(): index for index, name in enumerate(calendar.month_name) if name
}
_FED = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_BEA_ARCHIVE = "https://www.bea.gov/news/archive"
_BEA_SCHEDULE = "https://www.bea.gov/news/schedule/full"
_ISM = (
    "https://www.ismworld.org/supply-management-news-and-reports/"
    "reports/rob-report-calendar"
)
_ISM_PRESS = "https://www.prnewswire.com/news/institute-for-supply-management/"
_MICHIGAN = "https://data.sca.isr.umich.edu/technical-docs.php"
_OPTIONS = "https://www.optionseducation.org/referencelibrary/expiration-calendar"
_NYSE = "https://www.nyse.com/markets/hours-calendars"
_NEW_YORK = ZoneInfo("America/New_York")


class EventCalendarError(ValueError):
    """The requested event year cannot be established from a complete source."""


def canonical_event_type(value: str) -> str:
    """Accept event names case-insensitively, ignoring spaces and underscores."""
    if isinstance(value, str):
        key = re.sub(r"[\s_]", "", value).casefold()
        for event in EVENT_TYPES:
            if re.sub(r"[\s_]", "", event).casefold() == key:
                return event
    raise ValueError(f"Unknown event type: {value!r}; expected one of {EVENT_TYPES}")


@dataclass(frozen=True)
class EventYear:
    """A provider's explicit declaration of complete calendar-year coverage."""

    event_type: str
    year: int
    dates: tuple[date, ...]
    sources: tuple[str, ...]
    complete: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous coverage, malformed dates, and missing provenance."""
        canonical = canonical_event_type(self.event_type)
        if canonical != self.event_type:
            raise EventCalendarError("EventYear must use the canonical event name")
        if isinstance(self.year, bool) or not isinstance(self.year, int):
            raise EventCalendarError("Event year must be an integer")
        if not 1900 <= self.year <= 9998:
            raise EventCalendarError("Event year must be between 1900 and 9998")
        if self.complete is not True:
            raise EventCalendarError(
                f"Incomplete calendar for {canonical}, {self.year}"
            )
        if not self.sources or any(
            not isinstance(item, str) or not item.strip() for item in self.sources
        ):
            raise EventCalendarError("An event calendar requires source provenance")
        if any(
            not isinstance(item, date)
            or isinstance(item, datetime)
            or item.year != self.year
            for item in self.dates
        ):
            raise EventCalendarError(f"Event dates must belong to {self.year}")
        if tuple(sorted(set(self.dates))) != self.dates:
            raise EventCalendarError("Event dates must be sorted and unique")


def _month_number(value: str) -> int:
    """Resolve full or abbreviated English month names from published tables."""
    normalized = value.casefold().strip().rstrip(".")
    for name, number in _MONTHS.items():
        if normalized in (name, name[:3]):
            return number
    raise EventCalendarError(f"Unrecognized calendar month: {value!r}")


def _exchange_dates(event_type: str, year: int) -> EventYear:
    """Apply expiration rules to NYSE sessions, including historical closures."""
    sessions = _nyse_market_dates(date(year - 1, 12, 1), date(year + 1, 1, 31))[
        "date"
    ].to_list()
    dates = []
    months = [(year, month) for month in range(1, 13)]
    if event_type == "First Weekly":
        months.append((year + 1, 1))
    for nominal_year, month in months:
        if event_type in ("Triple Witching", "EndofQuarter") and month % 3:
            continue
        if event_type in ("EndofMonth", "EndofQuarter"):
            target = date(
                nominal_year, month, calendar.monthrange(nominal_year, month)[1]
            )
        else:
            first = date(nominal_year, month, 1)
            ordinal = 1 if event_type == "First Weekly" else 3
            target = first + timedelta(
                days=(4 - first.weekday()) % 7 + 7 * (ordinal - 1)
            )
        actual = max(session for session in sessions if session <= target)
        if actual.year == year:
            dates.append(actual)
    return EventYear(
        event_type, year, tuple(sorted(set(dates))), (_OPTIONS, _NYSE), True
    )


def _published_year(
    event_type: str, year: int, dates: list[date], *sources: str
) -> EventYear:
    """Create a complete year only after the source parser established coverage."""
    if not dates:
        raise EventCalendarError(f"No published dates for {event_type}, {year}")
    return EventYear(event_type, year, tuple(sorted(set(dates))), sources, True)


class OfficialEventProvider:
    """Fetch agency schedules once per document during a preparation operation.

    No economic release dates are inferred from monthly weekday patterns. A
    missing, blocked, or partial source raises instead of producing false flags.
    """

    def __init__(self) -> None:
        self._documents: dict[str, bytes] = {}
        self._sentiment_text: str | None = None

    def _download(self, url: str) -> bytes:
        """Download public documents with finite timeouts and shared reuse."""
        if url not in self._documents:
            try:
                response = requests.get(url, timeout=(10, 30))
                response.raise_for_status()
            except requests.RequestException as exc:
                raise EventCalendarError(
                    f"Cannot download event calendar {url}: {exc}"
                ) from exc
            self._documents[url] = response.content
        return self._documents[url]

    def _soup(self, url: str) -> BeautifulSoup:
        """Parse source HTML without running scripts."""
        return BeautifulSoup(self._download(url), "html.parser")

    def __call__(self, event_type: str, year: int) -> EventYear:
        event_type = canonical_event_type(event_type)
        if event_type in _RULE_EVENTS:
            return _exchange_dates(event_type, year)
        if event_type == "FOMC Meeting":
            return self._fomc(year)
        if event_type in ("CPI Release", "PPI Release", "Nonfarm Payrolls"):
            return self._bls(event_type, year)
        if event_type == "PCE Release":
            return self._bea(year)
        if event_type in ("Manufacturing PMI", "Services PMI"):
            return self._ism(event_type, year)
        return self._sentiment(year)

    def _fomc(self, year: int) -> EventYear:
        """Use the final day of scheduled meetings, excluding cancelled meetings."""
        soup = self._soup(_FED)
        dates = []
        for panel in soup.select("div.panel"):
            heading = panel.find(re.compile(r"h[1-6]"))
            if (
                heading is None
                or heading.get_text(" ", strip=True) != f"{year} FOMC Meetings"
            ):
                continue
            for meeting in panel.select(".fomc-meeting"):
                month = meeting.select_one(".fomc-meeting__month")
                day = meeting.select_one(".fomc-meeting__date")
                if month is None or day is None:
                    raise EventCalendarError("FOMC meeting table changed format")
                label = day.get_text(" ", strip=True)
                if re.fullmatch(r"\d{1,2}(?:-\d{1,2})?\*?", label):
                    dates.append(
                        date(
                            year,
                            _month_number(month.get_text().split("/")[-1]),
                            int(re.findall(r"\d+", label)[-1]),
                        )
                    )
            break
        source = _FED
        if not dates:
            source = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
            soup = self._soup(source)
            for heading in soup.find_all(re.compile(r"h[1-6]")):
                label = heading.get_text(" ", strip=True)
                if "Meeting" not in label or any(
                    word in label.casefold()
                    for word in ("cancelled", "unscheduled", "conference")
                ):
                    continue
                match = re.fullmatch(
                    r"([A-Za-z]+)(?:/([A-Za-z]+))?\s+(\d{1,2})(?:-(\d{1,2}))? Meeting\s*[-–]\s*"
                    + str(year),
                    label,
                )
                if match:
                    month = _month_number(match[2] or match[1])
                    dates.append(date(year, month, int(match[4] or match[3])))
        if (
            not 6 <= len(dates) <= 10
            or (source == _FED and len(dates) != 8)
            or min(dates).month > 2
            or max(dates).month < 11
        ):
            raise EventCalendarError(
                f"No complete scheduled FOMC calendar for {year}: {source}"
            )
        return _published_year("FOMC Meeting", year, dates, source)

    def _bls(self, event_type: str, year: int) -> EventYear:
        """Prefer the agency schedule, with an official historical archive fallback."""
        try:
            return self._bls_published(event_type, year)
        except EventCalendarError:
            try:
                return self._fred_calendar(event_type, year)
            except EventCalendarError:
                if year >= datetime.now(_NEW_YORK).year:
                    raise
                return self._alfred(event_type, year)

    def _fred_calendar(self, event_type: str, year: int) -> EventYear:
        """Read the Fed's source-published calendar with explicit date filters."""
        release_id, release_name = {
            "CPI Release": (10, "Consumer Price Index"),
            "PPI Release": (46, "Producer Price Index"),
            "Nonfarm Payrolls": (50, "Employment Situation"),
            "PCE Release": (54, "Personal Income and Outlays"),
        }[event_type]
        start, end = date(year, 1, 1), date(year, 12, 31)
        source = (
            f"https://fred.stlouisfed.org/releases/calendar?rid={release_id}"
            f"&vs={start.isoformat()}&ve={end.isoformat()}"
        )
        soup = self._soup(source)
        fields = {}
        for field in ("vs", "ve", "min-rd", "max-rd"):
            element = soup.find("input", id=f"rc-{field}")
            if element is None:
                raise EventCalendarError(
                    f"FRED calendar has no coverage metadata: {source}"
                )
            try:
                fields[field] = date.fromisoformat(str(element.get("value")))
            except ValueError as exc:
                raise EventCalendarError(
                    f"Invalid FRED calendar coverage: {source}"
                ) from exc
        if (
            fields["vs"] != start
            or fields["ve"] != end
            or fields["min-rd"] > start
            or fields["max-rd"].year < year
        ):
            raise EventCalendarError(
                f"FRED calendar does not fully cover {year}: {source}"
            )
        results = soup.select_one("#release-dates-pager")
        if results is None:
            raise EventCalendarError(f"FRED calendar has no release table: {source}")
        pager = re.search(
            r"Releases\s+(\d+)\s*-\s*(\d+)\s+of\s+(\d+)",
            results.get_text(" ", strip=True),
        )
        if pager is None or int(pager[1]) != 1 or pager[2] != pager[3]:
            raise EventCalendarError(
                f"FRED calendar is incomplete or paginated: {source}"
            )
        dates = []
        published = None
        for row in results.find_all("tr"):
            label = row.find("span")
            if label is not None:
                match = re.fullmatch(
                    r"[A-Za-z]+\s+([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})",
                    label.get_text(" ", strip=True),
                )
                if match:
                    published = date(
                        int(match[3]), _month_number(match[1]), int(match[2])
                    )
            link = row.find("a", href=f"/release?rid={release_id}")
            if link is not None:
                if (
                    published is None
                    or published.year != year
                    or link.get_text(" ", strip=True) != release_name
                ):
                    raise EventCalendarError(
                        f"FRED calendar ignored its event/year filters: {source}"
                    )
                dates.append(published)
        if len(dates) != int(pager[3]):
            raise EventCalendarError(
                f"FRED calendar release count does not match: {source}"
            )
        return _published_year(event_type, year, dates, source)

    def _alfred(self, event_type: str, year: int) -> EventYear:
        """Read St. Louis Fed's published historical release dates, without keys.

        ALFRED includes dates on which any series in the release was revised.
        Its documented fallback, when publishers supply no date, is first data
        availability. Provenance identifies this fallback rather than implying
        dates were downloaded directly from the original agency.
        """
        release_id, release_name = {
            "CPI Release": (10, "Consumer Price Index"),
            "PPI Release": (46, "Producer Price Index"),
            "Nonfarm Payrolls": (50, "Employment Situation"),
            "PCE Release": (54, "Personal Income and Outlays"),
        }[event_type]
        source = f"https://alfred.stlouisfed.org/release/downloaddates?ff=txt&rid={release_id}"
        text = self._download(source).decode("utf-8-sig")
        if f"Release: {release_name}" not in text or "Release Dates:" not in text:
            raise EventCalendarError(f"Unrecognized ALFRED release archive: {source}")
        created = re.search(r"File Created:\s+(\d{4}-\d{2}-\d{2})", text)
        dates = [
            date.fromisoformat(value)
            for value in re.findall(
                r"^\d{4}-\d{2}-\d{2}$", text.split("Release Dates:", 1)[1], re.MULTILINE
            )
        ]
        if (
            not dates
            or created is None
            or date.fromisoformat(created[1]).year <= year
            or min(dates).year >= year
        ):
            raise EventCalendarError(f"ALFRED does not fully cover {year}: {source}")
        return _published_year(
            event_type, year, [item for item in dates if item.year == year], source
        )

    def _bls_published(self, event_type: str, year: int) -> EventYear:
        """Read all twelve agency calendar months, including schedule revisions."""
        source = f"https://www.bls.gov/schedule/{year}/home.htm"
        soup = self._soup(source)
        months = {
            heading.get_text(" ", strip=True)
            for heading in soup.find_all(re.compile(r"h[1-6]"))
        }
        if any(f"{month} {year}" not in months for month in calendar.month_name[1:]):
            raise EventCalendarError(
                f"BLS calendar does not cover all months of {year}: {source}"
            )
        release = {
            "CPI Release": "Consumer Price Index",
            "PPI Release": "Producer Price Index",
            "Nonfarm Payrolls": "Employment Situation",
        }[event_type]
        dates = []
        covered_months = set()
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue
            match = re.search(
                r"([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})",
                cells[0].get_text(" ", strip=True),
            )
            if match and int(match[3]) == year:
                published = date(year, _month_number(match[1]), int(match[2]))
                covered_months.add(published.month)
                if release in row.get_text(" ", strip=True):
                    dates.append(published)
        if covered_months != set(range(1, 13)):
            raise EventCalendarError(f"BLS calendar rows do not cover {year}: {source}")
        return _published_year(event_type, year, dates, source)

    def _bea_archive(self, year: int) -> EventYear | None:
        """Read actual publication dates for years indexed by the BEA archive."""
        if year >= datetime.now(_NEW_YORK).year:
            return None
        archive = self._soup(_BEA_ARCHIVE)
        selector = archive.find("select", attrs={"name": "created_1"})
        if isinstance(selector, Tag):
            option = next(
                (
                    item
                    for item in selector.find_all("option")
                    if item.get_text(strip=True) == str(year)
                ),
                None,
            )
            if option is not None:
                source = (
                    _BEA_ARCHIVE
                    + "?"
                    + urlencode(
                        {
                            "field_related_product_target_id": "All",
                            "created_1": option.get("value"),
                            "title": "Personal Income and Outlays",
                        }
                    )
                )
                soup = self._soup(source)
                if soup.select(".pager-next, .pager__item--next"):
                    raise EventCalendarError(
                        f"BEA archive is paginated for {year}: {source}"
                    )
                dates = []
                for row in soup.find_all("tr"):
                    timestamp = row.find("time")
                    if (
                        timestamp is not None
                        and "Personal Income and Outlays,"
                        in row.get_text(" ", strip=True)
                    ):
                        parsed = datetime.fromisoformat(
                            str(timestamp.get("datetime"))
                        ).date()
                        if parsed.year != year:
                            raise EventCalendarError(
                                "BEA archive ignored its year filter"
                            )
                        dates.append(parsed)
                return _published_year("PCE Release", year, dates, source)
        return None

    def _bea(self, year: int) -> EventYear:
        """Use published archive dates or the agency's complete current schedule."""
        try:
            archived = self._bea_archive(year)
        except EventCalendarError:
            archived = None
        if archived is not None:
            return archived
        try:
            return self._fred_calendar("PCE Release", year)
        except EventCalendarError:
            if year < datetime.now(_NEW_YORK).year:
                return self._alfred("PCE Release", year)
        return self._bea_schedule(year)

    def _bea_schedule(self, year: int) -> EventYear:
        """Require a complete agency schedule when no exact Fed calendar exists."""
        soup = self._soup(_BEA_SCHEDULE)
        source = _BEA_SCHEDULE
        for link in soup.select("nav.tabs a"):
            if link.get_text(" ", strip=True) == f"{year} Schedule":
                source = urljoin(_BEA_SCHEDULE, str(link.get("href")))
                soup = self._soup(source)
                break
        years = {
            int(match[1])
            for link in soup.find_all("a", href=True)
            if (match := re.search(r"/news/(\d{4})/", str(link["href"])))
        }
        table = soup.select_one("#release-schedule-table")
        if table is None or years != {year}:
            raise EventCalendarError(f"No complete BEA schedule for {year}: {source}")
        dates = []
        months = set()
        for row in table.find_all("tr"):
            day = row.select_one(".release-date")
            title = row.select_one(".release-title")
            if day is None or title is None:
                continue
            match = re.fullmatch(
                r"([A-Za-z]+)\s+(\d{1,2})", day.get_text(" ", strip=True)
            )
            if match:
                month = _month_number(match[1])
                months.add(month)
                if title.get_text(" ", strip=True).startswith(
                    "Personal Income and Outlays,"
                ):
                    dates.append(date(year, month, int(match[2])))
        if months != set(range(1, 13)):
            raise EventCalendarError(f"BEA schedule is incomplete for {year}: {source}")
        return _published_year("PCE Release", year, dates, source)

    def _ism(self, event_type: str, year: int) -> EventYear:
        """Prefer ISM's published calendar, then a complete recorded-history year."""
        try:
            return self._ism_published(event_type, year)
        except EventCalendarError:
            try:
                return self._ism_press_history(event_type, year)
            except EventCalendarError:
                return self._ism_history(event_type, year)

    def _ism_press_page(
        self, url: str, event_type: str
    ) -> tuple[list[tuple[date, str]], date]:
        """Identify monthly reports and exact publication dates in ISM's archive."""
        soup = self._soup(url)
        publications = []
        reports = []
        for link in soup.select("a.newsreleaseconsolidatelink"):
            heading = link.find("h3")
            timestamp = None if heading is None else heading.find("small")
            if timestamp is None:
                continue
            match = re.match(
                r"([A-Za-z]{3})\s+(\d{1,2}),\s+(\d{4}),",
                timestamp.get_text(" ", strip=True),
            )
            if match is None:
                raise EventCalendarError(
                    f"Unrecognized ISM publication timestamp: {url}"
                )
            published = date(int(match[3]), _month_number(match[1]), int(match[2]))
            publications.append(published)
            href = str(link.get("href", ""))
            monthly = re.search(r"(?:pmi|nmi)-at-\d", href) is not None
            services = "non-manufacturing" in href or "services" in href
            matches = (
                services
                if event_type == "Services PMI"
                else "manufacturing" in href and not services
            )
            if monthly and matches:
                reports.append((published, urljoin(url, href)))
        if not publications:
            raise EventCalendarError(f"No ISM press archive entries at {url}")
        return reports, min(publications)

    def _ism_press_history(self, event_type: str, year: int) -> EventYear:
        """Read ISM's own press releases, preserving report/actual-date distinction."""
        dates: set[date] = set()
        sources = []
        latest_date = date.min
        latest_url = ""
        previous = date.max
        for page in range(1, 21):
            source = _ISM_PRESS + f"?page={page}&pagesize=100"
            reports, oldest = self._ism_press_page(source, event_type)
            sources.append(source)
            if oldest >= previous:
                raise EventCalendarError(
                    f"ISM press archive ignored pagination: {source}"
                )
            previous = oldest
            for published, report in reports:
                if published.year == year:
                    dates.add(published)
                    if published > latest_date:
                        latest_date, latest_url = published, report
            if oldest < date(year, 1, 1):
                break
        if len(dates) == 11 and year == datetime.now(_NEW_YORK).year and latest_url:
            report = self._soup(latest_url).get_text(" ", strip=True)
            upcoming = re.search(
                r"next ISM.{0,500}?will be released.{0,200}?\bon\s+"
                r"(?:[A-Za-z]+,\s*)?([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})",
                report,
                flags=re.IGNORECASE,
            )
            if upcoming:
                announced = date(
                    int(upcoming[3]), _month_number(upcoming[1]), int(upcoming[2])
                )
                if announced.year == year:
                    dates.add(announced)
                    sources.append(latest_url)
        if len(dates) != 12 or {item.month for item in dates} != set(range(1, 13)):
            raise EventCalendarError(
                f"ISM press history is incomplete for {event_type}, {year}"
            )
        return _published_year(event_type, year, list(dates), *sources)

    def _ism_history(self, event_type: str, year: int) -> EventYear:
        """Accept recorded third-party dates only with all twelve release months."""
        indicator = (
            "us-ism-manufacturing"
            if event_type == "Manufacturing PMI"
            else "us-ism-services"
        )
        source = f"https://makkler.com/labs/economic-calendar/{indicator}"
        soup = self._soup(source)
        dates: set[date] = set()

        def collect(value: object) -> None:
            if isinstance(value, dict):
                if value.get("indicator") == indicator and isinstance(
                    value.get("at"), str
                ):
                    parsed = datetime.fromisoformat(
                        value["at"].replace("Z", "+00:00")
                    ).date()
                    if parsed.year == year:
                        dates.add(parsed)
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        for script in soup.find_all("script"):
            match = re.fullmatch(
                r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', script.get_text()
            )
            if match is None:
                continue
            for line in json.loads(match[1]).splitlines():
                body = line.partition(":")[2]
                if body.startswith(("[", "{")):
                    try:
                        decoded = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    collect(decoded)
        if len(dates) != 12 or {item.month for item in dates} != set(range(1, 13)):
            raise EventCalendarError(
                f"Recorded ISM history is incomplete for {event_type}, {year}: {source}"
            )
        return _published_year(event_type, year, list(dates), source)

    def _ism_published(self, event_type: str, year: int) -> EventYear:
        """Require the published yearly table; do not infer ISM-specific holidays."""
        soup = self._soup(_ISM)
        dates = []
        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            match = re.fullmatch(
                r"([A-Za-z]+)\s+" + str(year), cells[0].get_text(" ", strip=True)
            )
            if match:
                selected = cells[1 if event_type == "Manufacturing PMI" else 2]
                match_day = re.match(r"\d{1,2}", selected.get_text(" ", strip=True))
                if match_day:
                    dates.append(date(year, _month_number(match[1]), int(match_day[0])))
        if {item.month for item in dates} != set(range(1, 13)):
            raise EventCalendarError(
                f"ISM does not publish a complete accessible {year} calendar at {_ISM}"
            )
        return _published_year(event_type, year, dates, _ISM)

    def _sentiment(self, year: int) -> EventYear:
        """Include both published preliminary and final sentiment release dates."""
        soup = self._soup(_MICHIGAN)
        link = next(
            (
                item
                for item in soup.find_all("a", href=True)
                if item.get_text(" ", strip=True)
                == "Historical Prelim/Final Release Dates"
            ),
            None,
        )
        if link is None:
            raise EventCalendarError(
                f"No historical sentiment schedule linked at {_MICHIGAN}"
            )
        source = urljoin(_MICHIGAN, str(link["href"]))
        if self._sentiment_text is None:
            reader = PdfReader(BytesIO(self._download(source)))
            self._sentiment_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        text = self._sentiment_text
        coverage = re.search(r"(\d{4})\s+through\s+(\d{4})", text)
        if coverage is None or not int(coverage[1]) < year <= int(coverage[2]):
            raise EventCalendarError(
                f"Consumer sentiment schedule does not fully cover {year}: {source}"
            )
        rows = re.findall(
            r"[A-Za-z]{3}\s+(\d{4})\s+(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}/\d{1,2}/\d{4})",
            text,
        )
        if sum(int(row[0]) == year for row in rows) != 12:
            raise EventCalendarError(
                f"Consumer sentiment calendar is incomplete for {year}: {source}"
            )
        dates = [
            parsed
            for _, prelim, final in rows
            for value in (prelim, final)
            if (parsed := datetime.strptime(value, "%m/%d/%Y").date()).year == year
        ]
        return _published_year("Consumer Sentiment", year, dates, source)
