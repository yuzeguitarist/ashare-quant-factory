import pandas as pd

from ashare_quant_factory.config import Data, Email, Paths, Project, Risk, Schedule, Settings, Backtest, GA
from ashare_quant_factory.data.fetcher import DataFetcher


class DummyDB:
    pass


def _settings() -> Settings:
    return Settings(
        project=Project(),
        paths=Paths(),
        data=Data(history_start="2020-01-01", probe_symbol="sh.000001"),
        schedule=Schedule(),
        watchlist=("sh.600000",),
        backtest=Backtest(),
        ga=GA(),
        risk=Risk(),
        email=Email(),
    )


def test_normalize_bars_keeps_expected_columns():
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "code": "sh.600000",
                "open": "10.1",
                "high": "10.2",
                "low": "10.0",
                "close": "10.15",
                "volume": "10000",
                "amount": "100000",
                "turn": "0.5",
                "tradestatus": "1",
                "pctChg": "1.2",
                "isST": "0",
            }
        ]
    )
    out = DataFetcher._normalize_bars(df)
    assert set(["code", "date", "open", "close", "pctChg"]).issubset(out.columns)
    assert out.loc[0, "open"] == 10.1


def test_next_date_when_empty_uses_history_start():
    f = DataFetcher(_settings(), DummyDB())
    assert f._next_date(None) == "2020-01-01"
    assert f._next_date("2020-01-01") == "2020-01-02"
