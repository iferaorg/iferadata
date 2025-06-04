import datetime
import holidays
import holidays.financial
from typing import Optional
from .enums import ExpirationRule

# -----------------------------------------------------------------------------
# 1. ENUMERATE ALL RULE TYPES
# -----------------------------------------------------------------------------

# ExpirationRule enum has been moved to enums.py

# -----------------------------------------------------------------------------
# 2. MAP FUTURES-MONTH LETTERS TO CALENDAR MONTHS
# -----------------------------------------------------------------------------

FUTURES_MONTH_CODES = {
    'F': 1,    # January
    'G': 2,    # February
    'H': 3,    # March
    'J': 4,    # April
    'K': 5,    # May
    'M': 6,    # June
    'N': 7,    # July
    'Q': 8,    # August
    'U': 9,    # September
    'V': 10,   # October
    'X': 11,   # November
    'Z': 12,   # December
}


# -----------------------------------------------------------------------------
# 3. HOLIDAY CALENDARS
# -----------------------------------------------------------------------------

# US market holidays:
US_HOLIDAYS = holidays.financial.ny_stock_exchange.NewYorkStockExchange()

# UK (England) public holidays:
UK_HOLIDAYS = holidays.financial_holidays('UK', subdiv='ENG')


# -----------------------------------------------------------------------------
# 4. BASIC BUSINESS-DAY HELPERS
# -----------------------------------------------------------------------------

def is_business_day(dt: datetime.date,
                    *,
                    country: str = 'US') -> bool:
    """
    Returns True if `dt` is a business day in the specified country’s calendar.
    Weekends are always non–business days; holidays come from python‐holidays.
    """
    if dt.weekday() >= 5:
        return False  # Saturday or Sunday

    if country.upper() == 'US':
        return dt not in US_HOLIDAYS
    elif country.upper() == 'UK':
        return dt not in UK_HOLIDAYS
    else:
        # fallback: treat all weekdays as business days
        return True


def adjust_to_previous_business_day(
    dt: datetime.date,
    *,
    country: str = 'US'
) -> datetime.date:
    """
    If `dt` is not a business day (in given country), walk backwards
    until a business day is found.
    """
    d = dt
    while not is_business_day(d, country=country):
        d -= datetime.timedelta(days=1)
    return d


def prior_month(
    year: int,
    month: int
) -> tuple[int, int]:
    """
    Return the year and month of the month prior to (year, month).
    Handles year rollover correctly.
    """
    if month == 1:
        return year - 1, 12
    else:
        return year, month - 1


def last_business_day_of_month(
    year: int,
    month: int,
    *,
    country: str = 'US'
) -> datetime.date:
    """
    Return the last business day (Mon–Fri excluding holidays) of a given month.
    """
    # Find the last calendar day of that month:
    if month == 12:
        last_day = datetime.date(year, 12, 31)
    else:
        last_day = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)

    return adjust_to_previous_business_day(last_day, country=country)


def third_last_business_day_of_month(
    year: int,
    month: int,
    *,
    country: str = 'US'
) -> datetime.date:
    """
    Return the third-last business day of that month.
    """
    # Starting from last business day, step backwards counting business days:
    count = 0
    d = last_business_day_of_month(year, month, country=country)
    while True:
        if is_business_day(d, country=country):
            count += 1
            if count == 3:
                return d
        d -= datetime.timedelta(days=1)


def nth_business_day_of_month(
    year: int,
    month: int,
    n: int,
    *,
    country: str = 'US'
) -> datetime.date:
    """
    Return the nth business day of the given month (1-based).
    """
    d = datetime.date(year, month, 1)
    count = 0
    while True:
        if is_business_day(d, country=country):
            count += 1
            if count == n:
                return d
        d += datetime.timedelta(days=1)


def business_days_before(
    ref_date: datetime.date,
    n: int,
    *,
    country: str = 'US'
) -> datetime.date:
    """
    Return the date that is `n` business days before `ref_date`.
    For example, if n=1, returns the prior business day.
    """
    d = ref_date - datetime.timedelta(days=1)
    count = 0
    while True:
        if is_business_day(d, country=country):
            count += 1
            if count == n:
                return d
        d -= datetime.timedelta(days=1)


def nth_weekday_of_month(
    year: int,
    month: int,
    weekday: int,
    n: int
) -> datetime.date:
    """
    Return the date of the nth occurrence of `weekday` in that month.
    `weekday`: Monday=0 … Sunday=6. `n` is 1-based.
    E.g. for 3rd Friday: weekday=4, n=3.
    """
    # Start from the first day of the month:
    d = datetime.date(year, month, 1)
    count = 0
    while True:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += datetime.timedelta(days=1)
        if d.month != month:
            raise ValueError(f"Month {month}/{year} has fewer than {n} occurrences of weekday {weekday}.")


# -----------------------------------------------------------------------------
# 5. RULE-BY-RULE IMPLEMENTATIONS
# -----------------------------------------------------------------------------

def _rule_last_business_day_contract_month(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates on the last business day of the contract month."""
    return last_business_day_of_month(year, month, country='US')


def _rule_third_last_business_day_contract_month(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates on the third-last business day of the contract month."""
    return third_last_business_day_of_month(year, month, country='US')


def _rule_seven_business_days_prior_last_bus_day(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates 7 business days prior to the last business day of the contract month."""
    last_bd = last_business_day_of_month(year, month, country='US')
    return business_days_before(last_bd, 7, country='US')


def _rule_tenth_business_day_contract_month(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates on the 10th business day of the contract month."""
    return nth_business_day_of_month(year, month, 10, country='US')


def _rule_last_business_day_prior_month(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates on the last business day of the month prior to the contract month."""
    prev_year, prev_month = prior_month(year, month)
    return last_business_day_of_month(prev_year, prev_month, country='US')


def _rule_third_last_business_day_prior_month(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates on the third-last business day of the month prior to the contract month."""
    prev_year, prev_month = prior_month(year, month)
    return third_last_business_day_of_month(prev_year, prev_month, country='US')


def _rule_third_friday_contract_month(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates on the 3rd Friday of the contract month."""
    raw = nth_weekday_of_month(year, month, weekday=4, n=3)  # Friday=4
    return adjust_to_previous_business_day(raw, country='US')


def _rule_one_bus_day_prior_third_wed(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates 1 business day prior to the third Wednesday of the contract month."""
    third_wed = nth_weekday_of_month(year, month, weekday=2, n=3)  # Wednesday=2
    raw = business_days_before(third_wed, 1, country='US')
    return raw  # already a business day


def _rule_two_bus_days_prior_third_wed(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates 2 business days prior to the third Wednesday of the contract month."""
    third_wed = nth_weekday_of_month(year, month, weekday=2, n=3)  # Wednesday=2
    raw = business_days_before(third_wed, 2, country='US')
    return raw


def _rule_three_bus_days_before_25th_prior_month(
    year: int,
    month: int
) -> datetime.date:
    """
    Trading terminates 3 business days before the 25th calendar day of the month prior.
    If the 25th is NOT a business day, trading terminates 4 business days before the 25th
    of the prior month.
    """
    # Determine prior‐month and year:
    prev_year, prev_month = prior_month(year, month)

    d25 = datetime.date(prev_year, prev_month, 25)
    if is_business_day(d25, country='US'):
        # 3 business days before:
        return business_days_before(d25, 3, country='US')
    else:
        # 4 business days before:
        return business_days_before(d25, 4, country='US')


def _rule_four_bus_days_before_25th_prior_month(
    year: int,
    month: int
) -> datetime.date:
    """
    Trading terminates 4 business days before the 25th calendar day of the month prior.
    If the 25th is NOT a business day, trading terminates 5 business days before the 25th.
    """
    # Determine prior‐month and year:
    prev_year, prev_month = prior_month(year, month)

    d25 = datetime.date(prev_year, prev_month, 25)
    if is_business_day(d25, country='US'):
        return business_days_before(d25, 4, country='US')
    else:
        return business_days_before(d25, 5, country='US')


def _rule_business_day_prior_15th(
    year: int,
    month: int
) -> datetime.date:
    """Trading terminates on the business day prior to the 15th day of the contract month."""
    d15 = datetime.date(year, month, 15)
    return business_days_before(d15, 1, country='US')


def _rule_feb_london_special(
    year: int,
    month: int
) -> datetime.date:
    """
    - If contract month is February: trading terminates on the 2nd-last London business day
      of (month – 2).
    - Otherwise: trading terminates on the last London business day of (month – 2).
    """
    prev_year, prev_month = prior_month(*prior_month(year, month)) # 2 months prior

    if month == 2:
        # 2nd-last London business day of prev_month:
        last_ld = last_business_day_of_month(prev_year, prev_month, country='UK')
        # step backwards to find the 2nd last:
        return business_days_before(last_ld, 1, country='UK')
    else:
        # last London business day of prev_month:
        return last_business_day_of_month(prev_year, prev_month, country='UK')


def _rule_last_thursday_special(
    year: int,
    month: int
) -> datetime.date:
    """
    Trading terminates on the last Thursday of the contract month, subject to:
      - If that Thursday or any of the four prior weekdays 
        is NOT a business day, move back by 7 days and re‐check.
      - Exception: For November contracts, terminates on the Thursday prior to
        Thanksgiving Day (which is the 4th Thursday of Nov). Then apply the
        same “preceded by a business day” rule.
    """
    def _find_last_thursday(y: int, m: int) -> datetime.date:
        # Find the last Thursday of (y,m):
        # Start from the last day of month, go backward until weekday == Thursday(3).
        if m == 12:
            cand = datetime.date(y, 12, 31)
        else:
            cand = datetime.date(y, m + 1, 1) - datetime.timedelta(days=1)
        while cand.weekday() != 3:  # Thursday = 3
            cand -= datetime.timedelta(days=1)
        return cand

    def _check_previous_n_weekdays(cand: datetime.date, n: int, country_code: str) -> bool:
        """
        Check if the given date `cand` and the prior `n` calendar weekdays
        are all business days in the specified country.
        """
        if not is_business_day(cand, country=country_code):
            return False
        
        needed = n
        d = cand - datetime.timedelta(days=1)

        while needed > 0:
            if d.weekday() < 5:
                if not is_business_day(d, country=country_code):
                    return False
                needed -= 1
            d -= datetime.timedelta(days=1)
        return True

    def _previous_clean_thursday(cand: datetime.date, country_code: str) -> datetime.date:
        """
        Given a Thursday candidate, check if it AND the prior 4 calendar‐day weekdays
        are all business days. If not, step back by 7 days and try again, cheking only
        the previous one weekday.
        """
        if not _check_previous_n_weekdays(cand, 4, country_code):
            cand -= datetime.timedelta(days=7)
            # Step back by 7 days and recheck:
            while not _check_previous_n_weekdays(cand, 1, country_code):
                cand -= datetime.timedelta(days=7)
                
        return cand

    # If November rule:
    if month == 11:
        # Thanksgiving = 4th Thursday of November:
        thanksgiving = nth_weekday_of_month(year, 11, weekday=3, n=4)
        cand = thanksgiving - datetime.timedelta(days=7)  # Thursday prior
        return _previous_clean_thursday(cand, country_code='US')

    # For all other months:
    raw_last_thu = _find_last_thursday(year, month)
    return _previous_clean_thursday(raw_last_thu, country_code='US')


def _rule_thursday_prior_second_friday_con_month(
    year: int,
    month: int
) -> datetime.date:
    """
    Trading terminates on the Thursday prior to the second Friday of the contract month.
    """
    # Find the second Friday of the month:
    second_friday = nth_weekday_of_month(year, month, weekday=4, n=2)  # Friday=4
    # Step back to the prior Thursday:
    raw_thursday = second_friday - datetime.timedelta(days=1)

    # Check if that Thursday is a business day:
    return adjust_to_previous_business_day(raw_thursday, country='US')


# -----------------------------------------------------------------------------
# 6. DISPATCHER FUNCTION
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
}


def calculate_expiration(
    month_code: str,
    rule: ExpirationRule,
    asset_class: Optional[str] = None
) -> datetime.date:
    """
    Given a futures-month code (e.g. "G15" meaning Feb 2015) and a rule,
    return the expiration date according to that rule.

    Assumes two-digit year → 2000 + year. Adjust if needed for other centuries.
    """
    # 1) Parse the code:
    if len(month_code) < 2:
        raise ValueError(f"Invalid month code: {month_code}")

    mon_letter = month_code[0].upper()
    year_digits = month_code[1:]
    if mon_letter not in FUTURES_MONTH_CODES:
        raise ValueError(f"Unknown month letter in code: {mon_letter}")

    month = FUTURES_MONTH_CODES[mon_letter]
    yy = int(year_digits)
    year = 2000 + yy  # NOTE: adjust logic if you need a century‐rollover rule

    # 2) Dispatch to the correct rule implementation:
    handler = _RULE_HANDLERS.get(rule)
    if handler:
        return handler(year, month)
    else:
        raise NotImplementedError(f"Rule {rule} not implemented.")

    # Columbus Day and Veterans Day closures for FX and Interest Rate futures
    if asset_class in ['FX', 'Interest Rate']:
        if rule == ExpirationRule.LAST_BUSINESS_DAY_CONTRACT_MONTH:
            if datetime.date(year, month, 1).month == 10:  # October
                return business_days_before(datetime.date(year, month, 1), 3, country='US')
            elif datetime.date(year, month, 1).month == 11:  # November
                return business_days_before(datetime.date(year, month, 1), 2, country='US')
