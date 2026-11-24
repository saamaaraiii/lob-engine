import numpy as np

from lob.simulation import MarketMakerParams, run_simulation


def test_simulation_is_deterministic():
    a = run_simulation(2_000, mm_params=MarketMakerParams(), seed=7)
    b = run_simulation(2_000, mm_params=MarketMakerParams(), seed=7)
    np.testing.assert_array_equal(a.pnl, b.pnl)


def test_pnl_accounting():
    r = run_simulation(3_000, mm_params=MarketMakerParams(), seed=3, keep_trades=True)
    inv = cash = 0
    for t in r.trades:
        if t.buyer == "mm":
            inv += t.quantity
            cash -= t.price * t.quantity
        if t.seller == "mm":
            inv -= t.quantity
            cash += t.price * t.quantity
    assert inv == r.inventory[-1]
    assert cash == r.cash[-1]


def test_inventory_skew_reduces_inventory_risk():
    skew = [run_simulation(10_000, mm_params=MarketMakerParams(), seed=s) for s in range(3)]
    naive = [
        run_simulation(10_000, mm_params=MarketMakerParams(inventory_skew=False), seed=s)
        for s in range(3)
    ]
    assert np.mean([r.inventory.std() for r in skew]) < np.mean([r.inventory.std() for r in naive])


def test_book_stays_two_sided():
    r = run_simulation(5_000, seed=1)
    assert np.isnan(r.spread).mean() < 0.01
    assert np.nanmin(r.spread) >= 1
