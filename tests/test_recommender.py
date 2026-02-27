import pandas as pd

from ashare_quant_factory.strategy.backtester import CostModel, Genome
from ashare_quant_factory.strategy.recommender import make_recommendations


def _make_price_df(n: int = 80) -> pd.DataFrame:
    rows = []
    price = 10.0
    for i in range(n):
        date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
        open_px = price * 1.001
        close_px = price * 1.002
        high_px = max(open_px, close_px) * 1.002
        low_px = min(open_px, close_px) * 0.998
        rows.append(
            {
                'date': date.strftime('%Y-%m-%d'),
                'open': open_px,
                'high': high_px,
                'low': low_px,
                'close': close_px,
                'tradestatus': '1',
            }
        )
        price = close_px
    return pd.DataFrame(rows)


def test_make_recommendations_keeps_empty_symbol_and_alignment():
    dataset = {
        'sh.600000': pd.DataFrame(),
        'sh.600001': _make_price_df(),
    }
    genome = Genome(
        fast_ma=5,
        slow_ma=30,
        rsi_period=14,
        rsi_buy=30,
        rsi_sell=70,
        atr_period=14,
        stop_loss_atr=2.0,
        take_profit_atr=4.0,
    )
    costs = CostModel()

    recos = make_recommendations(dataset, genome, costs, max_weight_per_symbol=0.5)

    assert set(recos['code']) == {'sh.600000', 'sh.600001'}
    empty_row = recos.loc[recos['code'] == 'sh.600000'].iloc[0]
    assert empty_row['action'] == 'HOLD'
    assert empty_row['weight'] == 0.0
    assert empty_row['note'] == 'No data in DB'
