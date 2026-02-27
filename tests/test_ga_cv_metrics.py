import pandas as pd

from ashare_quant_factory.strategy.backtester import CostModel, Genome
from ashare_quant_factory.strategy.genetic import evaluate_genome


def _make_df(n: int = 180) -> pd.DataFrame:
    rows = []
    px = 10.0
    for i in range(n):
        dt = pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)
        drift = 1.0008 if i % 20 < 12 else 0.9995
        open_px = px * 1.0005
        close_px = px * drift
        high_px = max(open_px, close_px) * 1.002
        low_px = min(open_px, close_px) * 0.998
        rows.append({'date': dt.strftime('%Y-%m-%d'), 'open': open_px, 'high': high_px, 'low': low_px, 'close': close_px, 'tradestatus': '1'})
        px = close_px
    return pd.DataFrame(rows)


def test_evaluate_genome_walk_forward_outputs_stability_and_sensitivity():
    dataset = {'sh.600000': _make_df(), 'sz.000001': _make_df()}
    g = Genome(5, 30, 14, 30, 70, 14, 2.0, 4.0)
    res = evaluate_genome(dataset, g, CostModel(), cv_mode='walk_forward', cv_splits=3, cv_purge_days=2)

    assert res.metrics['cv_mode'] == 'walk_forward'
    assert res.metrics['cv_fold_count'] > 0
    assert 0 <= res.metrics['stability_score'] <= 100
    assert isinstance(res.metrics['param_sensitivity'], list)
    assert len(res.metrics['param_sensitivity']) >= 1
