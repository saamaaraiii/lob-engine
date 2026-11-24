"""Order, trade and execution-report types.

Prices are stored as integer *ticks* so that matching never depends on
floating-point comparisons. Convert to currency at the edges of the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(str, Enum):
    GTC = "gtc"  # good-till-cancelled: remainder rests on the book
    IOC = "ioc"  # immediate-or-cancel: fill what you can, cancel the rest
    FOK = "fok"  # fill-or-kill: fill completely right now or not at all


class OrderStatus(str, Enum):
    NEW = "new"  # rested on the book with no fills
    PARTIALLY_FILLED = "partially_filled"  # some fills, remainder resting
    FILLED = "filled"
    CANCELLED = "cancelled"  # unfilled remainder cancelled (IOC / market)
    REJECTED = "rejected"  # FOK that could not be filled in full


@dataclass(slots=True)
class Order:
    order_id: int
    side: Side
    quantity: int
    price: int | None = None
    order_type: OrderType = OrderType.LIMIT
    tif: TimeInForce = TimeInForce.GTC
    owner: str = ""
    remaining: int = field(init=False)
    timestamp: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type is OrderType.LIMIT:
            if self.price is None or self.price <= 0:
                raise ValueError("limit orders need a positive integer price in ticks")
        elif self.price is not None:
            raise ValueError("market orders must not have a price")
        self.remaining = self.quantity

    @property
    def filled(self) -> int:
        return self.quantity - self.remaining


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: int
    price: int
    quantity: int
    aggressor_side: Side
    buy_order_id: int
    sell_order_id: int
    buyer: str
    seller: str
    timestamp: int


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    order_id: int
    status: OrderStatus
    trades: tuple[Trade, ...]
    remaining: int

    @property
    def filled_quantity(self) -> int:
        return sum(t.quantity for t in self.trades)

    @property
    def average_price(self) -> float | None:
        qty = self.filled_quantity
        if qty == 0:
            return None
        return sum(t.price * t.quantity for t in self.trades) / qty
