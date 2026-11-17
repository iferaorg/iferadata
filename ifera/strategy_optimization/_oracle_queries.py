"""Fixed Oracle 21c queries for the inspected IFERARO minute-data schema.

The raw Greeks primary key is (SYMBOL, DATETIME, EXP_DATE, STRIKE, RIGHT).
Both queries receive one JSON array through the ``requests`` CLOB bind. Request
timestamps use ISO dates with six fractional-second digits. No catalog reads,
source guessing, user-defined functions, or per-trade SQL executions are needed.
"""

UNDERLYING_SQL = """
WITH requests AS (
    SELECT j.symbol,
           CAST(j.start_at AS DATE) AS start_at,
           CAST(j.end_at AS DATE) AS end_at
    FROM JSON_TABLE(:requests, '$[*]' COLUMNS (
        symbol VARCHAR2(10) PATH '$.symbol' ERROR ON ERROR,
        start_at TIMESTAMP PATH '$.start' ERROR ON ERROR,
        end_at TIMESTAMP PATH '$.end' ERROR ON ERROR
    )) j
), minute_offsets AS (
    SELECT LEVEL - 1 AS minute_offset
    FROM dual
    CONNECT BY LEVEL <= 391
), requested_minutes AS (
    SELECT /*+ NO_MERGE(r) LEADING(r s m n) USE_NL(s m) INDEX(m OPT_MARKET_DATE_MV_PK) */
           r.symbol AS underlying,
           s.symbol AS option_symbol,
           CAST(m.t_date + (570 + n.minute_offset) / 1440 AS TIMESTAMP) AS stamp
    FROM requests r
    JOIN OPT_SYMBOL s ON s.underlying_symbol = r.symbol
    JOIN OPT_MARKET_DATE_MV m ON m.symbol = s.symbol
                           AND m.t_date >= r.start_at
                           AND m.t_date < r.end_at
    CROSS JOIN minute_offsets n
)
SELECT /*+ NO_PARALLEL */ r.underlying AS symbol,
       r.stamp AS datetime,
       (SELECT /*+ NO_UNNEST INDEX(g TDO_GREEKS_ALL_1M_RAW_PK) */ g.underlying_price
        FROM TDO_GREEKS_ALL_1M_RAW g
        WHERE g.symbol = r.option_symbol
          AND g.datetime = r.stamp
          AND g.underlying_price > 0
          AND ROWNUM = 1) AS price
FROM requested_minutes r
ORDER BY r.underlying, r.stamp
"""


OPTION_SQL = """
WITH request_times AS (
    SELECT j.trade_index,
           j.symbol,
           j.direction,
           j.option_right,
           j.strike,
           j.expiration,
           j.start_at,
           j.end_at - j.start_at AS duration
    FROM JSON_TABLE(:requests, '$[*]' COLUMNS (
        trade_index NUMBER PATH '$.trade_index' ERROR ON ERROR,
        symbol VARCHAR2(10) PATH '$.symbol' ERROR ON ERROR,
        direction VARCHAR2(5) PATH '$.direction' ERROR ON ERROR,
        option_right VARCHAR2(1) PATH '$.right' ERROR ON ERROR,
        strike NUMBER PATH '$.strike' ERROR ON ERROR,
        expiration DATE PATH '$.expiration' ERROR ON ERROR,
        start_at TIMESTAMP PATH '$.start' ERROR ON ERROR,
        end_at TIMESTAMP PATH '$.end' ERROR ON ERROR
    )) j
), requests AS (
    SELECT r.*,
           GREATEST(1, CEIL(EXTRACT(DAY FROM r.duration) * 1440
                         + EXTRACT(HOUR FROM r.duration) * 60
                         + EXTRACT(MINUTE FROM r.duration)
                         + EXTRACT(SECOND FROM r.duration) / 60)) AS minute_count
    FROM request_times r
), minute_offsets AS (
    SELECT LEVEL - 1 AS minute_offset
    FROM dual
    CONNECT BY LEVEL <= 1440
), requested_minutes AS (
    SELECT r.trade_index,
           r.direction,
           r.option_right,
           r.strike,
           r.expiration,
           s.symbol AS option_symbol,
           r.start_at + NUMTODSINTERVAL(n.minute_offset, 'MINUTE') AS stamp
    FROM requests r
    JOIN OPT_SYMBOL s ON s.underlying_symbol = r.symbol
    JOIN minute_offsets n ON n.minute_offset < r.minute_count
)
SELECT /*+ NO_PARALLEL LEADING(r) USE_NL(g) INDEX(g TDO_GREEKS_ALL_1M_RAW_PK) */
       r.trade_index,
       g.datetime,
       r.direction,
       CASE r.option_right WHEN 'C' THEN 'call' ELSE 'put' END AS right,
       g.strike,
       g.bid,
       g.ask,
       (g.bid + g.ask) / 2 AS mid,
       g.delta
FROM requested_minutes r
JOIN TDO_GREEKS_ALL_1M_RAW g ON g.symbol = r.option_symbol
                         AND g.datetime = r.stamp
                         AND g.exp_date = r.expiration
                         AND g.strike = r.strike
                         AND g.right = r.option_right
ORDER BY r.trade_index, g.datetime, r.direction, r.option_right, g.strike
"""
