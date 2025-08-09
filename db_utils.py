from __future__ import annotations
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Iterable, Optional, Union

# ---- Engine ----
def get_engine(db_url: str = "sqlite:///market_data.db"):
    """Return a SQLAlchemy engine."""
    return create_engine(db_url, future=True)

# ---- Generic IO ----
def write_df(df: pd.DataFrame, table: str, db_url: str, if_exists: str = "append"):
    """Dump a DataFrame to SQL."""
    eng = get_engine(db_url)
    with eng.begin() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)

def list_tables(db_url: str) -> list[str]:
    """List tables in the DB."""
    eng = get_engine(db_url)
    with eng.connect() as conn:
        if db_url.startswith("sqlite"):
            rows = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        else:  # postgres, etc.
            rows = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
        return [r[0] for r in rows]

# ---- Symbols & timestamps ----
def get_symbols(db_url: str) -> list[str]:
    """All symbols in price_data."""
    eng = get_engine(db_url)
    with eng.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT symbol FROM price_data ORDER BY symbol"))
        return [r[0] for r in rows]

def last_price_timestamp(db_url: str, symbol: str) -> Optional[pd.Timestamp]:
    """Latest Date for a symbol."""
    eng = get_engine(db_url)
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT MAX([Date]) FROM price_data WHERE symbol = :s"), {"s": symbol}
        ).fetchone()
    ts = row[0]
    return pd.to_datetime(ts) if ts is not None else None

# ---- Prices ----
def read_prices(
    db_url: str,
    symbols: Optional[Iterable[str]] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    price_col: str = "Adj Close",
    wide: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV from price_data. Default: adjusted close.
    Set wide=True to pivot symbols into columns.
    """
    eng = get_engine(db_url)
    where = ["1=1"]
    params = {}
    if symbols:
        where.append("symbol IN :syms")
        params["syms"] = tuple(symbols)
    if start:
        where.append("[Date] >= :st")
        params["st"] = pd.to_datetime(start)
    if end:
        where.append("[Date] <= :en")
        params["en"] = pd.to_datetime(end)

    sql = f"""
    SELECT [Date] AS dt, symbol, "Open","High","Low","Close","Adj Close","Volume"
    FROM price_data
    WHERE {' AND '.join(where)}
    ORDER BY dt, symbol
    """
    df = pd.read_sql_query(sql, eng, params=params, parse_dates=["dt"])
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not in columns: {list(df.columns)}")

    if wide:
        out = df.pivot(index="dt", columns="symbol", values=price_col).sort_index()
        out.index.name = "date"
        return out
    return df.rename(columns={"dt": "date"}).set_index("date")

def resample_prices(prices_wide: pd.DataFrame, rule: str = "B") -> pd.DataFrame:
    """
    Resample wide prices to a new freq. Forward-fill between periods.
    Good for aligning to business days, hours, etc.
    """
    return prices_wide.resample(rule).last().ffill()

# ---- Macro ----
def read_macro(
    db_url: str,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Load macro_data (one row per date, columns = FRED series)."""
    eng = get_engine(db_url)
    where = []
    params = {}
    if start:
        where.append("date >= :st")
        params["st"] = pd.to_datetime(start)
    if end:
        where.append("date <= :en")
        params["en"] = pd.to_datetime(end)

    sql = "SELECT * FROM macro_data"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY date"
    df = pd.read_sql_query(sql, eng, params=params, parse_dates=["date"]).set_index("date").sort_index()
    # cast numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---- Joins ----
def join_prices_macro(
    prices_wide: pd.DataFrame, macro_df: pd.DataFrame, how: str = "left"
) -> pd.DataFrame:
    """Join wide price matrix with macro series on date index."""
    out = prices_wide.join(macro_df, how=how)
    # forward-fill macro (monthly) to match daily bars
    out[macro_df.columns] = out[macro_df.columns].ffill()
    return out

# ---- Quick checks ----
def head(db_url: str, table: str, n: int = 5) -> pd.DataFrame:
    """Peek at a table."""
    eng = get_engine(db_url)
    return pd.read_sql_query(f"SELECT * FROM {table} LIMIT {n}", eng)

if __name__ == "__main__":
    DB = "sqlite:///market_data.db"
    print("Tables:", list_tables(DB))
    print("Symbols:", get_symbols(DB))
    px = read_prices(DB, symbols=None, start="2020-01-01", price_col="Adj Close", wide=True)
    mc = read_macro(DB, start="2020-01-01")
    joined = join_prices_macro(px, mc)
    print(px.head())
    print(mc.head())
    print(joined.head())
