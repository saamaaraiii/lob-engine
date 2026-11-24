"""Throughput and latency benchmark for the matching engine.

Pre-generates a realistic mix of order flow (limit orders near the touch,
market orders and cancels), then times every engine call individually with
``time.perf_counter_ns``.

    python benchmarks/bench.py --orders 500000
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from lob import Order, OrderBook, OrderType, Side


def generate(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    kinds = rng.choice(3, size=n, p=[0.60, 0.10, 0.30])
    sides = rng.integers(0, 2, size=n)
    qtys = rng.integers(1, 100, size=n)
    offsets = rng.geometric(0.15, size=n)
    return kinds, sides, qtys, offsets, rng


def run(n: int) -> dict[str, float]:
    kinds, sides, qtys, offsets, rng = generate(n)
    kinds, sides, qtys, offsets = (a.tolist() for a in (kinds, sides, qtys, offsets))
    picks = rng.random(n).tolist()
    book = OrderBook("BENCH")
    mid = 10_000
    live: list[int] = []
    lat = np.empty(n, dtype=np.int64)
    t_start = time.perf_counter()
    for i in range(n):
        oid = i + 1
        side = Side.BUY if sides[i] else Side.SELL
        kind = kinds[i]
        if kind == 0:
            px = mid - offsets[i] + 1 if side is Side.BUY else mid + offsets[i] - 1
            order = Order(oid, side, int(qtys[i]), int(px))
            t0 = time.perf_counter_ns()
            book.submit(order)
            lat[i] = time.perf_counter_ns() - t0
            live.append(oid)
        elif kind == 1:
            order = Order(oid, side, int(qtys[i]), order_type=OrderType.MARKET)
            t0 = time.perf_counter_ns()
            book.submit(order)
            lat[i] = time.perf_counter_ns() - t0
        else:
            target = live[int(picks[i] * len(live))] if live else 0
            t0 = time.perf_counter_ns()
            book.cancel(target)
            lat[i] = time.perf_counter_ns() - t0
        if book.mid is not None:
            mid = int(book.mid)
        if len(live) > 2 * len(book) + 10_000:
            live = [o for o in live if o in book]
    wall = time.perf_counter() - t_start
    return {
        "orders": n,
        "engine_ops_per_sec": n / (lat.sum() / 1e9),
        "end_to_end_ops_per_sec": n / wall,
        "p50_us": np.percentile(lat, 50) / 1e3,
        "p99_us": np.percentile(lat, 99) / 1e3,
        "p999_us": np.percentile(lat, 99.9) / 1e3,
        "resting_orders_at_end": len(book),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=500_000)
    args = ap.parse_args()
    res = run(args.orders)
    print(f"Python {platform.python_version()} on {platform.machine()} ({platform.system()})")
    for k, v in res.items():
        print(f"{k:>24}: {v:,.2f}" if isinstance(v, float) else f"{k:>24}: {v:,}")
