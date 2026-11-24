"""Compare an Avellaneda-Stoikov market maker with a naive symmetric quoter.

Runs both strategies on the same simulated order flow (same random seeds) and
saves the charts used in the README to ``docs/``.

    python examples/market_making.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lob.simulation import MarketMakerParams, run_simulation

OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(exist_ok=True)
STEPS, SEEDS = 20_000, range(10)


def main() -> None:
    rows = {}
    runs = {}
    for name, skew in (("Avellaneda-Stoikov", True), ("Naive symmetric", False)):
        res = [
            run_simulation(STEPS, mm_params=MarketMakerParams(inventory_skew=skew), seed=s)
            for s in SEEDS
        ]
        runs[name] = res[0]
        summ = [r.summary() for r in res]
        rows[name] = {k: np.mean([s[k] for s in summ]) for k in summ[0]}

    print(f"Averages over {len(SEEDS)} seeds x {STEPS:,} steps\n")
    print(f"| Metric | {' | '.join(rows)} |")
    print("|---|" + "---:|" * len(rows))
    for key, label in [
        ("pnl_per_1k_steps", "PnL per 1k steps (ticks)"),
        ("sharpe_per_1k_steps", "Sharpe (per 1k steps)"),
        ("inventory_std", "Inventory std (shares)"),
        ("max_abs_inventory", "Max abs inventory (shares)"),
        ("mm_volume", "Volume traded (shares)"),
    ]:
        print(f"| {label} | " + " | ".join(f"{rows[n][key]:,.2f}" for n in rows) + " |")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    a = runs["Avellaneda-Stoikov"]
    axes[0].plot(a.mid / 100, color="#333", lw=0.8)
    axes[0].set_ylabel("Mid price ($)")
    axes[0].set_title("Simulated market (zero-intelligence order flow)")
    for name, colour in (("Avellaneda-Stoikov", "#1f77b4"), ("Naive symmetric", "#d62728")):
        axes[1].plot(runs[name].inventory, label=name, color=colour, lw=0.8)
        axes[2].plot(runs[name].pnl / 100, label=name, color=colour, lw=0.8)
    axes[1].set_ylabel("Inventory (shares)")
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].legend(loc="upper left")
    axes[2].set_ylabel("Mark-to-market PnL ($)")
    axes[2].set_xlabel("Event step")
    fig.tight_layout()
    fig.savefig(OUT / "market_making.png", dpi=130)

    # order-book depth snapshot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(a.spread[~np.isnan(a.spread)], bins=np.arange(0.5, 15.5, 1), color="#1f77b4")
    ax.set_xlabel("Bid-ask spread (ticks)")
    ax.set_ylabel("Steps")
    ax.set_title("Spread distribution")
    fig.tight_layout()
    fig.savefig(OUT / "spread_distribution.png", dpi=130)
    print(f"\nCharts saved to {OUT}")


if __name__ == "__main__":
    main()
