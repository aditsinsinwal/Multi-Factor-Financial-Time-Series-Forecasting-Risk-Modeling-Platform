import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import requests

#config
TICKERS = ["AAPL", "MSFT", "GOOG", "TSLA", "SPY"]  # assets to fetch
DB_PATH = "sqlite:///market_data.db" # SQLite DB file
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

#FRED macro data
FRED_API_KEY =   #get your own from https://fred.stlouisfed.org/
FRED_SERIES = {
    "UNRATE": "Unemployment Rate",  # US Unemployment Rate
    "DGS10": "10-Year Treasury Constant Maturity Rate"
}



#Helper fxns
def fetch_yf_data(tickers, start, end):
   #Download OHLCV data for multiple tickers from Yahoo Finance
    all_data = {}
    for t in tickers:
        print(f"Fetching {t}...")
        df = yf.download(t, start=start, end=end)
        df = df.reset_index()
        df["symbol"] = t
        all_data[t] = df
    combined = pd.concat(all_data.values(), ignore_index=True)
    return combined


def fetch_fred_data(series_dict, start, end, api_key):
   #Fetch macro series from FRED API.
    if api_key is None:
        print("No FRED API key provided. Skipping macro fetch.")
        return pd.DataFrame()

    all_series = []
    for code, name in series_dict.items():
        print(f"Fetching macro series: {code} ({name})")
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": code,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()["observations"]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.rename(columns={"value": code}, inplace=True)
        df = df[["date", code]]
        all_series.append(df)
    # merge all macro series on date
    from functools import reduce
    macro_df = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), all_series)
    return macro_df


def save_to_db(df, table_name, db_path):
    """Save DataFrame to SQL table."""
    engine = create_engine(db_path)
    df.to_sql(table_name, engine, if_exists="append", index=False)
    engine.dispose()

#MAIN
if __name__ == "__main__":
    #1. Market data
    market_df = fetch_yf_data(TICKERS, START_DATE, END_DATE)
    save_to_db(market_df, "price_data", DB_PATH)
    print(f"Saved {len(market_df)} rows of market data to DB.")

    #2. Macro data (optional)
    if FRED_API_KEY:
        macro_df = fetch_fred_data(FRED_SERIES, START_DATE, END_DATE, FRED_API_KEY)
        save_to_db(macro_df, "macro_data", DB_PATH)
        print(f"Saved {len(macro_df)} rows of macro data to DB.")
    else:
        print("Macro data skipped.")
