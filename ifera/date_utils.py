"""
Date and time utilities for financial instruments.

This module provides functions for calculating futures contract expiration dates
and related date operations based on various market-specific rules.
"""

import datetime
from typing import Optional

from holidays import financial_holidays
from holidays.financial import ny_stock_exchange
from .enums import ExpirationRule

# -----------------------------------------------------------------------------
# 1. ENUMERATE ALL RULE TYPES
# -----------------------------------------------------------------------------

# ExpirationRule enum has been moved to enums.py

# -----------------------------------------------------------------------------
# 2. MAP FUTURES-MONTH LETTERS TO CALENDAR MONTHS
# -----------------------------------------------------------------------------

FUTURES_MONTH_CODES = {
    "F": 1,  # January
    "G": 2,  # February
    "H": 3,  # March
    "J": 4,  # April
    "K": 5,  # May
    "M": 6,  # June
    "N": 7,  # July
    "Q": 8,  # August
    "U": 9,  # September
    "V": 10,  # October
    "X": 11,  # November
    "Z": 12,  # December
}

# -----------------------------------------------------------------------------
# 3. HOLIDAY CALENDARS
# -----------------------------------------------------------------------------

# US market holidays:
US_HOLIDAYS = ny_stock_exchange.NewYorkStockExchange()

# UK (England) public holidays:
UK_HOLIDAYS = financial_holidays("UK", subdiv="ENG")

# -----------------------------------------------------------------------------
# 4. ADDITIONAL HOLIDAY CHECKS
# -----------------------------------------------------------------------------


def is_columbus_day(dt: datetime.date) -> bool:
    """Check if the date is Columbus Day (second Monday in October)."""
    return dt.month == 10 and dt.weekday() == 0 and 8 <= dt.day <= 14


def is_veterans_day_observed(dt: datetime.date) -> bool:
    """Check if the date is Veterans Day observed."""
    if dt.month != 11:
        return False
    nov11 = datetime.date(dt.year, 11, 11)
    if nov11.weekday() < 5:  # Monday to Friday
        return dt == nov11
    elif nov11.weekday() == 5:  # Saturday
        return dt == nov11 - datetime.timedelta(days=1)  # Friday
    elif nov11.weekday() == 6:  # Sunday
        return dt == nov11 + datetime.timedelta(days=1)  # Monday
    return False


# -----------------------------------------------------------------------------
# 5. BASIC BUSINESS-DAY HELPERS
# -----------------------------------------------------------------------------


def is_business_day(
    dt: datetime.date, *, country: str = "US", asset_class: Optional[str] = None
) -> bool:
    """
    Returns True if `dt` is a business day in the specified country’s calendar.
    Weekends are always non-business days; holidays include standard financial holidays,
    plus Columbus Day and Veterans Day for FX and Interest Rate futures.
    """
    if dt.weekday() >= 5:
        return False
    if country.upper() == "US":
        if dt in US_HOLIDAYS:
            return False
        if asset_class in ["FX", "Interest Rate"]:
            if is_columbus_day(dt) or is_veterans_day_observed(dt):
                return False
        return True
    elif country.upper() == "UK":
        return dt not in UK_HOLIDAYS
    else:
        return True


def adjust_to_previous_business_day(
    dt: datetime.date, *, country: str = "US", asset_class: Optional[str] = None
) -> datetime.date:
    """Adjust to the previous business day if `dt` is not a business day."""
    d = dt
    while not is_business_day(d, country=country, asset_class=asset_class):
        d -= datetime.timedelta(days=1)
    return d


def prior_month(year: int, month: int) -> tuple[int, int]:
    """Return the year and month of the prior month."""
    if month == 1:
        return year - 1, 12
    else:
        return year, month - 1


def last_business_day_of_month(
    year: int, month: int, *, country: str = "US", asset_class: Optional[str] = None
) -> datetime.date:
    """Return the last business day of a given month."""
    if month == 12:
        last_day = datetime.date(year, 12, 31)
    else:
        last_day = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    return adjust_to_previous_business_day(
        last_day, country=country, asset_class=asset_class
    )


def third_last_business_day_of_month(
    year: int, month: int, *, country: str = "US", asset_class: Optional[str] = None
) -> datetime.date:
    """Return the third-last business day of that month."""
    count = 0
    d = last_business_day_of_month(
        year, month, country=country, asset_class=asset_class
    )
    while True:
        if is_business_day(d, country=country, asset_class=asset_class):
            count += 1
            if count == 3:
                return d
        d -= datetime.timedelta(days=1)


def nth_business_day_of_month(
    year: int,
    month: int,
    n: int,
    *,
    country: str = "US",
    asset_class: Optional[str] = None,
) -> datetime.date:
    """Return the nth business day of the given month (1-based)."""
    d = datetime.date(year, month, 1)
    count = 0
    while True:
        if is_business_day(d, country=country, asset_class=asset_class):
            count += 1
            if count == n:
                return d
        d += datetime.timedelta(days=1)


def business_days_before(
    ref_date: datetime.date,
    n: int,
    *,
    country: str = "US",
    asset_class: Optional[str] = None,
) -> datetime.date:
    """Return the date that is `n` business days before `ref_date`."""
    d = ref_date - datetime.timedelta(days=1)
    count = 0
    while True:
        if is_business_day(d, country=country, asset_class=asset_class):
            count += 1
            if count == n:
                return d
        d -= datetime.timedelta(days=1)


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> datetime.date:
    """Return the date of the nth occurrence of `weekday` in that month."""
    d = datetime.date(year, month, 1)
    count = 0
    while True:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += datetime.timedelta(days=1)
        if d.month != month:
            raise ValueError(
                f"Month {month}/{year} has fewer than {n} occurrences of weekday {weekday}."
            )


# -----------------------------------------------------------------------------
# 6. RULE-BY-RULE IMPLEMENTATIONS
# -----------------------------------------------------------------------------


def _rule_last_business_day_contract_month(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates on the last business day of the contract month."""
    return last_business_day_of_month(
        year, month, country="US", asset_class=asset_class
    )


def _rule_third_last_business_day_contract_month(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates on the third-last business day of the contract month."""
    return third_last_business_day_of_month(
        year, month, country="US", asset_class=asset_class
    )


def _rule_seven_business_days_prior_last_bus_day(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates 7 business days prior to the last business day of the contract month."""
    last_bd = last_business_day_of_month(
        year, month, country="US", asset_class=asset_class
    )
    return business_days_before(last_bd, 7, country="US", asset_class=asset_class)


def _rule_tenth_business_day_contract_month(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates on the 10th business day of the contract month."""
    return nth_business_day_of_month(
        year, month, 10, country="US", asset_class=asset_class
    )


def _rule_last_business_day_prior_month(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates on the last business day of the month prior to the contract month."""
    prev_year, prev_month = prior_month(year, month)
    return last_business_day_of_month(
        prev_year, prev_month, country="US", asset_class=asset_class
    )


def _rule_third_last_business_day_prior_month(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates on the third-last business day of the month prior to
    the contract month."""
    prev_year, prev_month = prior_month(year, month)
    return third_last_business_day_of_month(
        prev_year, prev_month, country="US", asset_class=asset_class
    )


def _rule_third_friday_contract_month(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates on the 3rd Friday of the contract month."""
    raw = nth_weekday_of_month(year, month, weekday=4, n=3)  # Friday=4
    return adjust_to_previous_business_day(raw, country="US", asset_class=asset_class)


def _rule_one_bus_day_prior_third_wed(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates 1 business day prior to the third Wednesday of the contract month."""
    third_wed = nth_weekday_of_month(year, month, weekday=2, n=3)  # Wednesday=2
    return business_days_before(third_wed, 1, country="US", asset_class=asset_class)


def _rule_two_bus_days_prior_third_wed(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates 2 business days prior to the third Wednesday of the contract month."""
    third_wed = nth_weekday_of_month(year, month, weekday=2, n=3)  # Wednesday=2
    return business_days_before(third_wed, 2, country="US", asset_class=asset_class)


def _rule_three_bus_days_before_25th_prior_month(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates 3 business days before the 25th of the prior month,
    or 4 if 25th is not a business day."""
    prev_year, prev_month = prior_month(year, month)
    d25 = datetime.date(prev_year, prev_month, 25)
    if is_business_day(d25, country="US", asset_class=asset_class):
        return business_days_before(d25, 3, country="US", asset_class=asset_class)
    else:
        return business_days_before(d25, 4, country="US", asset_class=asset_class)


def _rule_four_bus_days_before_25th_prior_month(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates 4 business days before the 25th of the prior month,
    or 5 if 25th is not a business day."""
    prev_year, prev_month = prior_month(year, month)
    d25 = datetime.date(prev_year, prev_month, 25)
    if is_business_day(d25, country="US", asset_class=asset_class):
        return business_days_before(d25, 4, country="US", asset_class=asset_class)
    else:
        return business_days_before(d25, 5, country="US", asset_class=asset_class)


def _rule_business_day_prior_15th(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates on the business day prior to the 15th day of the contract month."""
    d15 = datetime.date(year, month, 15)
    return business_days_before(d15, 1, country="US", asset_class=asset_class)


def _rule_bz_pre2016(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """
    Compute the last trading day for Brent Last Day Financial Futures
    under the pre-2016 rule (business day prior to the 15th calendar day
    before the contract month).
    """
    contract_start = datetime.date(year, month, 1)
    d15 = contract_start - datetime.timedelta(days=15)
    last_trading_day = business_days_before(
        d15, 1, country="UK", asset_class=asset_class
    )

    return last_trading_day


def _rule_feb_london_special(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Special rule for February and other months using London business days."""
    # ----------  Historical cut-over  ----------
    if (year < 2016) or (year == 2016 and month <= 2):
        return _rule_bz_pre2016(year, month, asset_class=asset_class)

    prev_year, prev_month = prior_month(*prior_month(year, month))
    if month == 2:
        last_ld = last_business_day_of_month(
            prev_year, prev_month, country="UK", asset_class=asset_class
        )
        return business_days_before(last_ld, 1, country="UK", asset_class=asset_class)
    else:
        return last_business_day_of_month(
            prev_year, prev_month, country="UK", asset_class=asset_class
        )


def _rule_last_thursday_special(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates on the last Thursday with specific conditions."""

    def _find_last_thursday(y: int, m: int) -> datetime.date:
        if m == 12:
            cand = datetime.date(y, 12, 31)
        else:
            cand = datetime.date(y, m + 1, 1) - datetime.timedelta(days=1)
        while cand.weekday() != 3:  # Thursday = 3
            cand -= datetime.timedelta(days=1)
        return cand

    def _check_previous_n_weekdays(
        cand: datetime.date, n: int, country_code: str
    ) -> bool:
        if not is_business_day(cand, country=country_code, asset_class=asset_class):
            return False
        needed = n
        d = cand - datetime.timedelta(days=1)
        while needed > 0:
            if d.weekday() < 5:
                if not is_business_day(
                    d, country=country_code, asset_class=asset_class
                ):
                    return False
                needed -= 1
            d -= datetime.timedelta(days=1)
        return True

    def _previous_clean_thursday(
        cand: datetime.date, country_code: str
    ) -> datetime.date:
        if not _check_previous_n_weekdays(cand, 4, country_code):
            cand -= datetime.timedelta(days=7)
            while not _check_previous_n_weekdays(cand, 1, country_code):
                cand -= datetime.timedelta(days=7)
        return cand

    if month == 11:
        thanksgiving = nth_weekday_of_month(year, 11, weekday=3, n=4)
        cand = thanksgiving - datetime.timedelta(days=7)
        return _previous_clean_thursday(cand, country_code="US")
    raw_last_thu = _find_last_thursday(year, month)
    return _previous_clean_thursday(raw_last_thu, country_code="US")


def _rule_thursday_prior_second_friday_con_month(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """Trading terminates on the Thursday prior to the second Friday of the contract month."""
    second_friday = nth_weekday_of_month(year, month, weekday=4, n=2)  # Friday=4
    raw_thursday = second_friday - datetime.timedelta(days=1)
    return adjust_to_previous_business_day(
        raw_thursday, country="US", asset_class=asset_class
    )


def _rule_first_business_day_after_first_friday(
    year: int, month: int, asset_class: Optional[str] = None
) -> datetime.date:
    """
    “First business day after the first Friday of the contract month.”
    1. Find the first Friday in that month/year.
    2. Advance one calendar day, then loop forward until you hit a business day.
    """
    # 1. Compute the first Friday (weekday=4) of the given month/year:
    first_friday = nth_weekday_of_month(year, month, weekday=4, n=1)

    # 2. Move to the next calendar day, then skip weekends/holidays:
    d = first_friday + datetime.timedelta(days=1)
    while not is_business_day(d, country="US", asset_class=asset_class):
        d += datetime.timedelta(days=1)

    return d


# -----------------------------------------------------------------------------
# 7. DISPATCHER FUNCTION
# -----------------------------------------------------------------------------

_RULE_HANDLERS = {
    ExpirationRule.LAST_BUSINESS_DAY_CONTRACT_MONTH: _rule_last_business_day_contract_month,
    ExpirationRule.THIRD_LAST_BUSINESS_DAY_CONTRACT_MONTH: _rule_third_last_business_day_contract_month,
    ExpirationRule.SEVEN_BUSINESS_DAYS_PRIOR_LAST_BUS_DAY: _rule_seven_business_days_prior_last_bus_day,
    ExpirationRule.TENTH_BUSINESS_DAY_CONTRACT_MONTH: _rule_tenth_business_day_contract_month,
    ExpirationRule.LAST_BUSINESS_DAY_PRIOR_MONTH: _rule_last_business_day_prior_month,
    ExpirationRule.THIRD_LAST_BUSINESS_DAY_PRIOR_MONTH: _rule_third_last_business_day_prior_month,
    ExpirationRule.THIRD_FRIDAY_CONTRACT_MONTH: _rule_third_friday_contract_month,
    ExpirationRule.ONE_BUSINESS_DAY_PRIOR_THIRD_WEDNESDAY: _rule_one_bus_day_prior_third_wed,
    ExpirationRule.TWO_BUSINESS_DAYS_PRIOR_THIRD_WEDNESDAY: _rule_two_bus_days_prior_third_wed,
    ExpirationRule.THREE_BUSINESS_DAYS_BEFORE_25TH_PRIOR_MONTH: _rule_three_bus_days_before_25th_prior_month,
    ExpirationRule.FOUR_BUSINESS_DAYS_BEFORE_25TH_PRIOR_MONTH: _rule_four_bus_days_before_25th_prior_month,
    ExpirationRule.BUSINESS_DAY_PRIOR_15TH: _rule_business_day_prior_15th,
    ExpirationRule.FEB_LONDON_SPECIAL_RULE: _rule_feb_london_special,
    ExpirationRule.LAST_THURSDAY_SPECIAL_RULE: _rule_last_thursday_special,
    ExpirationRule.THURSDAY_PRIOR_SECOND_FRIDAY_CON_MONTH: _rule_thursday_prior_second_friday_con_month,
    ExpirationRule.FIRST_BUSINESS_DAY_AFTER_FIRST_FRIDAY: _rule_first_business_day_after_first_friday,
}


def calculate_expiration(
    month_code: str, rule: ExpirationRule, asset_class: Optional[str] = None
) -> datetime.date:
    """
    Given a futures-month code (e.g. "G15" meaning Feb 2015) and a rule,
    return the expiration date according to that rule, adjusted for asset class holidays.
    """
    if len(month_code) < 2:
        raise ValueError(f"Invalid month code: {month_code}")

    mon_letter = month_code[0].upper()
    year_digits = month_code[1:]
    if mon_letter not in FUTURES_MONTH_CODES:
        raise ValueError(f"Unknown month letter in code: {mon_letter}")

    month = FUTURES_MONTH_CODES[mon_letter]
    yy = int(year_digits)
    year = 2000 + yy

    handler = _RULE_HANDLERS.get(rule)
    if handler:
        return handler(year, month, asset_class=asset_class)
    else:
        raise NotImplementedError(f"Rule {rule} not implemented.")
