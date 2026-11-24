"""Central limit order book with price-time priority matching.

Data structures
---------------
* Each side is a ``SortedDict`` mapping price (ticks) -> ``PriceLevel``.
  Best bid is the last key of the bid side, best ask is the first key of the
  ask side, so finding the top of book is O(1) and inserting a new price
  level is O(log L) for L levels.
* Each ``PriceLevel`` keeps its resting orders in an ``OrderedDict`` keyed by
  order id. Insertion order gives time priority, and the dict gives O(1)
  cancellation of any order in the queue.
* A flat ``order_id -> Order`` index finds any resting order in O(1).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

from sortedcontainers import SortedDict

from .orders import (
    ExecutionReport,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    Trade,
)


class PriceLevel:
    __slots__ = ("price", "orders", "volume")

    def __init__(self, price: int) -> None:
        self.price = price
        self.orders: OrderedDict[int, Order] = OrderedDict()
        self.volume = 0

    def append(self, order: Order) -> None:
        self.orders[order.order_id] = order
        self.volume += order.remaining

    def __len__(self) -> int:
        return len(self.orders)


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    bids: list[tuple[int, int]]  # (price, volume), best first
    asks: list[tuple[int, int]]


class OrderBook:
    """A single-instrument limit order book and matching engine."""

    def __init__(self, symbol: str = "") -> None:
        self.symbol = symbol
        self._bids: SortedDict[int, PriceLevel] = SortedDict()
        self._asks: SortedDict[int, PriceLevel] = SortedDict()
        self._orders: dict[int, Order] = {}
        self._clock = 0
        self._next_trade_id = 1

    # ------------------------------------------------------------------ queries
    @property
    def best_bid(self) -> int | None:
        return self._bids.peekitem(-1)[0] if self._bids else None

    @property
    def best_ask(self) -> int | None:
        return self._asks.peekitem(0)[0] if self._asks else None

    @property
    def spread(self) -> int | None:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else ask - bid

    @property
    def mid(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else (bid + ask) / 2

    def get_order(self, order_id: int) -> Order | None:
        return self._orders.get(order_id)

    def __contains__(self, order_id: int) -> bool:
        return order_id in self._orders

    def __len__(self) -> int:
        """Number of resting orders."""
        return len(self._orders)

    def volume_at(self, side: Side, price: int) -> int:
        level = self._side(side).get(price)
        return level.volume if level else 0

    def depth(self, levels: int = 5) -> BookSnapshot:
        bids = [(p, lvl.volume) for p, lvl in self._iter_levels(Side.BUY)][:levels]
        asks = [(p, lvl.volume) for p, lvl in self._iter_levels(Side.SELL)][:levels]
        return BookSnapshot(bids=bids, asks=asks)

    def imbalance(self, levels: int = 1) -> float | None:
        """Order-book imbalance in [-1, 1] over the top ``levels`` levels."""
        snap = self.depth(levels)
        b = sum(v for _, v in snap.bids)
        a = sum(v for _, v in snap.asks)
        return None if a + b == 0 else (b - a) / (b + a)

    # ---------------------------------------------------------------- commands
    def submit(self, order: Order) -> ExecutionReport:
        """Match an incoming order and rest any GTC limit remainder."""
        if order.order_id in self._orders:
            raise ValueError(f"duplicate order id {order.order_id}")
        self._clock += 1
        order.timestamp = self._clock

        if order.tif is TimeInForce.FOK and not self._can_fill(order):
            return ExecutionReport(order.order_id, OrderStatus.REJECTED, (), order.remaining)

        trades = self._match(order)

        if order.remaining == 0:
            status = OrderStatus.FILLED
        elif order.order_type is OrderType.LIMIT and order.tif is TimeInForce.GTC:
            self._rest(order)
            status = OrderStatus.PARTIALLY_FILLED if trades else OrderStatus.NEW
        else:
            status = OrderStatus.CANCELLED
        return ExecutionReport(order.order_id, status, tuple(trades), order.remaining)

    def cancel(self, order_id: int) -> bool:
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        book = self._side(order.side)
        level = book[order.price]
        del level.orders[order_id]
        level.volume -= order.remaining
        if not level.orders:
            del book[order.price]
        return True

    def amend(
        self, order_id: int, new_quantity: int | None = None, new_price: int | None = None
    ) -> ExecutionReport | None:
        """Change a resting order.

        Reducing quantity at the same price keeps queue priority (this is how
        real exchanges behave). Increasing quantity or changing price is a
        cancel/replace: the order goes to the back of the queue and may trade
        immediately if the new price crosses.
        """
        order = self._orders.get(order_id)
        if order is None:
            return None
        qty = order.remaining if new_quantity is None else new_quantity
        price = order.price if new_price is None else new_price
        if qty <= 0:
            self.cancel(order_id)
            return ExecutionReport(order_id, OrderStatus.CANCELLED, (), 0)

        if price == order.price and qty <= order.remaining:
            level = self._side(order.side)[order.price]
            level.volume -= order.remaining - qty
            order.remaining = qty
            status = OrderStatus.NEW if order.filled == 0 else OrderStatus.PARTIALLY_FILLED
            return ExecutionReport(order_id, status, (), qty)

        self.cancel(order_id)
        replacement = Order(order_id, order.side, qty, price, owner=order.owner)
        return self.submit(replacement)

    # --------------------------------------------------------------- internals
    def _side(self, side: Side) -> SortedDict:
        return self._bids if side is Side.BUY else self._asks

    def _iter_levels(self, side: Side) -> Iterator[tuple[int, PriceLevel]]:
        """Levels from best to worst."""
        book = self._side(side)
        return reversed(book.items()) if side is Side.BUY else iter(book.items())

    @staticmethod
    def _crosses(order: Order, price: int) -> bool:
        if order.price is None:  # market order
            return True
        return price <= order.price if order.side is Side.BUY else price >= order.price

    def _can_fill(self, order: Order) -> bool:
        need = order.remaining
        for price, level in self._iter_levels(order.side.opposite):
            if not self._crosses(order, price):
                break
            need -= level.volume
            if need <= 0:
                return True
        return False

    def _match(self, order: Order) -> list[Trade]:
        trades: list[Trade] = []
        book = self._side(order.side.opposite)
        is_buy = order.side is Side.BUY
        while order.remaining and book:
            price, level = book.peekitem(0) if is_buy else book.peekitem(-1)
            if not self._crosses(order, price):
                break
            orders = level.orders
            while order.remaining and orders:
                resting = orders[next(iter(orders))]
                qty = min(order.remaining, resting.remaining)
                resting.remaining -= qty
                order.remaining -= qty
                level.volume -= qty
                buy, sell = (order, resting) if is_buy else (resting, order)
                trades.append(
                    Trade(
                        trade_id=self._next_trade_id,
                        price=price,  # trades print at the resting order's price
                        quantity=qty,
                        aggressor_side=order.side,
                        buy_order_id=buy.order_id,
                        sell_order_id=sell.order_id,
                        buyer=buy.owner,
                        seller=sell.owner,
                        timestamp=self._clock,
                    )
                )
                self._next_trade_id += 1
                if resting.remaining == 0:
                    orders.popitem(last=False)
                    del self._orders[resting.order_id]
            if not orders:
                del book[price]
        return trades

    def _rest(self, order: Order) -> None:
        book = self._side(order.side)
        level = book.get(order.price)
        if level is None:
            level = book[order.price] = PriceLevel(order.price)
        level.append(order)
        self._orders[order.order_id] = order
