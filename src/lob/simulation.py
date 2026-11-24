"""Agent-based market simulation on top of the matching engine.

Two kinds of participant trade on one ``OrderBook``:

* **Noise traders** following the zero-intelligence order-flow model of
  Farmer, Patelli & Zovko (2005): limit orders arrive at random distances
  from the opposite best price, market orders arrive at random, and resting
  orders are cancelled at random. This alone produces a realistic book shape
  and a diffusive mid price.
* **A market maker** quoting with the Avellaneda & Stoikov (2008) model. It
  skews its quotes away from its inventory using a reservation price

      r = s - q * gamma * sigma^2 * tau

  and quotes a spread of

      delta = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k)

  around it, where s is the mid, q the inventory (in lots), gamma risk
  aversion, sigma the mid volatility per step, tau the horizon and k the
  order-arrival decay. Switching ``inventory_skew`` off gives a naive
  symmetric quoter to compare against.

Everything is in integer ticks and integer shares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .book import OrderBook
from .orders import Order, OrderType, Side, Trade


@dataclass
class FlowParams:
    limit_prob: float = 0.50
    market_prob: float = 0.15
    cancel_prob: float = 0.35
    mean_size: float = 10.0
    offset_p: float = 0.05  # geometric parameter for distance from the best price
    initial_mid: int = 10_000
    initial_levels: int = 10


@dataclass
class MarketMakerParams:
    gamma: float = 0.1
    k: float = 1.0
    horizon: float = 50.0
    quote_size: int = 10
    max_inventory: int = 500
    requote_every: int = 5
    inventory_skew: bool = True
    vol_halflife: float = 200.0
    max_skew_ticks: float = 25.0  # cap on how far the reservation price moves from mid


@dataclass
class SimulationResult:
    mid: np.ndarray
    spread: np.ndarray
    inventory: np.ndarray
    cash: np.ndarray
    pnl: np.ndarray
    mm_volume: int
    n_trades: int
    trades: list[Trade] = field(repr=False, default_factory=list)

    def summary(self) -> dict[str, float]:
        step_pnl = np.diff(self.pnl)
        sd = step_pnl.std()
        return {
            "final_pnl_ticks": float(self.pnl[-1]),
            "pnl_per_1k_steps": float(step_pnl.mean() * 1000),
            "sharpe_per_1k_steps": float(step_pnl.mean() / sd * math.sqrt(1000)) if sd else 0.0,
            "inventory_std": float(self.inventory.std()),
            "max_abs_inventory": float(np.abs(self.inventory).max()),
            "mm_volume": float(self.mm_volume),
            "mean_spread_ticks": float(np.nanmean(self.spread)),
            "n_trades": float(self.n_trades),
        }


class AvellanedaStoikovMM:
    OWNER = "mm"

    def __init__(self, params: MarketMakerParams) -> None:
        self.p = params
        self.inventory = 0
        self.cash = 0.0
        self.volume = 0
        self.bid_id: int | None = None
        self.ask_id: int | None = None
        self._var = 1.0  # EWMA variance of per-step mid changes (ticks^2)
        self._alpha = 1 - 0.5 ** (1 / params.vol_halflife)
        self._last_mid: float | None = None

    def observe_mid(self, mid: float) -> None:
        if self._last_mid is not None:
            d = mid - self._last_mid
            self._var = (1 - self._alpha) * self._var + self._alpha * d * d
        self._last_mid = mid

    def on_trade(self, trade: Trade) -> None:
        if trade.buyer == self.OWNER:
            self.inventory += trade.quantity
            self.cash -= trade.price * trade.quantity
            self.volume += trade.quantity
        if trade.seller == self.OWNER:
            self.inventory -= trade.quantity
            self.cash += trade.price * trade.quantity
            self.volume += trade.quantity

    def quotes(self, mid: float) -> tuple[int, int]:
        p = self.p
        q = self.inventory / p.quote_size if p.inventory_skew else 0.0
        risk = p.gamma * self._var * p.horizon
        skew = max(-p.max_skew_ticks, min(p.max_skew_ticks, q * risk))
        reservation = mid - skew
        half = min(0.5 * risk, p.max_skew_ticks) + (1 / p.gamma) * math.log1p(p.gamma / p.k)
        return math.floor(reservation - half), math.ceil(reservation + half)

    def requote(self, book: OrderBook, next_id: int) -> tuple[int, list[Trade]]:
        """Cancel old quotes and post new passive ones. Returns the next free id."""
        for oid in (self.bid_id, self.ask_id):
            if oid is not None:
                book.cancel(oid)
        self.bid_id = self.ask_id = None
        mid = book.mid
        if mid is None:
            return next_id, []
        bid, ask = self.quotes(mid)
        # stay passive: never cross the spread with a quote
        if book.best_ask is not None:
            bid = min(bid, book.best_ask - 1)
        if book.best_bid is not None:
            ask = max(ask, book.best_bid + 1)
        trades: list[Trade] = []
        if self.inventory < self.p.max_inventory and bid > 0:
            rep = book.submit(Order(next_id, Side.BUY, self.p.quote_size, bid, owner=self.OWNER))
            trades += rep.trades
            if next_id in book:
                self.bid_id = next_id
            next_id += 1
        if self.inventory > -self.p.max_inventory:
            rep = book.submit(Order(next_id, Side.SELL, self.p.quote_size, ask, owner=self.OWNER))
            trades += rep.trades
            if next_id in book:
                self.ask_id = next_id
            next_id += 1
        return next_id, trades


def run_simulation(
    n_steps: int = 20_000,
    flow: FlowParams | None = None,
    mm_params: MarketMakerParams | None = None,
    seed: int = 0,
    keep_trades: bool = False,
) -> SimulationResult:
    flow = flow or FlowParams()
    rng = np.random.default_rng(seed)
    book = OrderBook("SIM")
    mm = AvellanedaStoikovMM(mm_params) if mm_params else None
    next_id = 1
    noise_ids: list[int] = []
    all_trades: list[Trade] = []
    n_trades = 0

    def size() -> int:
        return int(rng.geometric(1 / flow.mean_size))

    # seed the book so both sides have liquidity
    for i in range(1, flow.initial_levels + 1):
        for side, px in ((Side.BUY, flow.initial_mid - i), (Side.SELL, flow.initial_mid + i)):
            book.submit(Order(next_id, side, size() * 2, px, owner="noise"))
            noise_ids.append(next_id)
            next_id += 1

    mids = np.empty(n_steps)
    spreads = np.empty(n_steps)
    inv = np.zeros(n_steps)
    cash = np.zeros(n_steps)
    last_mid = float(flow.initial_mid)
    probs = np.array([flow.limit_prob, flow.market_prob, flow.cancel_prob])
    probs = probs / probs.sum()

    for t in range(n_steps):
        trades: list[Trade] = []
        if mm and t % mm.p.requote_every == 0:
            next_id, trades = mm.requote(book, next_id)

        event = rng.choice(3, p=probs)
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        if event == 0:  # limit order, placed behind the opposite best price
            offset = int(rng.geometric(flow.offset_p))
            if side is Side.BUY:
                ref = book.best_ask if book.best_ask is not None else math.ceil(last_mid)
                price = ref - offset
            else:
                ref = book.best_bid if book.best_bid is not None else math.floor(last_mid)
                price = ref + offset
            if price > 0:
                rep = book.submit(Order(next_id, side, size(), price, owner="noise"))
                trades += rep.trades
                if next_id in book:
                    noise_ids.append(next_id)
                next_id += 1
        elif event == 1:  # market order
            mkt = Order(next_id, side, size(), order_type=OrderType.MARKET, owner="noise")
            rep = book.submit(mkt)
            trades += rep.trades
            next_id += 1
        elif noise_ids:  # cancel a random resting noise order
            j = int(rng.integers(len(noise_ids)))
            noise_ids[j], noise_ids[-1] = noise_ids[-1], noise_ids[j]
            book.cancel(noise_ids.pop())

        # keep the id list from growing with already-filled orders
        if len(noise_ids) > 4 * max(len(book), 1):
            noise_ids = [i for i in noise_ids if i in book]

        if mm:
            for tr in trades:
                mm.on_trade(tr)
        n_trades += len(trades)
        if keep_trades:
            all_trades += trades

        m = book.mid
        if m is not None:
            last_mid = m
        mids[t] = last_mid
        spreads[t] = book.spread if book.spread is not None else np.nan
        if mm:
            mm.observe_mid(last_mid)
            inv[t] = mm.inventory
            cash[t] = mm.cash

    pnl = cash + inv * mids  # mark-to-market at the mid
    return SimulationResult(
        mid=mids,
        spread=spreads,
        inventory=inv,
        cash=cash,
        pnl=pnl,
        mm_volume=mm.volume if mm else 0,
        n_trades=n_trades,
        trades=all_trades,
    )
