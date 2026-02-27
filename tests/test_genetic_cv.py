import pandas as pd

from ashare_quant_factory.strategy.backtester import CostModel, Genome
from ashare_quant_factory.strategy.genetic import evaluate_genome, parameter_sensitivity


def _make_price_df(n: int = 260) -> pd.DataFrame:
    rows = []
    px = 10.0
    for i in range(n):
        d = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)
        drift = 1.0 + (0.001 if (i // 40) % 2 == 0 else -0.0005)
        open_px = px * (drift + 0.0002)
        close_px = px * drift
        high_px = max(open_px, close_px) * 1.002
        low_px = min(open_px, close_px) * 0.998
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "tradestatus": "1",
            }
        )
        px = close_px
    return pd.DataFrame(rows)


def _genome() -> Genome:
    return Genome(
        fast_ma=10,
        slow_ma=60,
        rsi_period=14,
        rsi_buy=35,
        rsi_sell=70,
        atr_period=14,
        stop_loss_atr=2.0,
        take_profit_atr=4.0,
    )


def test_walk_forward_adds_stability_metrics():
    dataset = {"sh.600519": _make_price_df(), "sz.000858": _make_price_df()}
    summary = evaluate_genome(dataset, _genome(), CostModel(), cv_method="walk_forward", cv_splits=3)

    assert "stability_score" in summary.metrics
    assert 0 <= int(summary.metrics["stability_score"]) <= 100
    assert summary.metrics["cv_method"] == "walk_forward"


def test_parameter_sensitivity_output_shape():
    dataset = {"sh.600519": _make_price_df()}
    rows = parameter_sensitivity(dataset, _genome(), CostModel(), cv_method="purged_cv", cv_splits=3)

    assert len(rows) == 8
    assert {"param", "base", "down", "up"}.issubset(rows[0].keys())
