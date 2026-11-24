"""Property-based differential testing.

Hypothesis generates random sequences of orders and cancels. Every sequence is
run through both the optimised ``OrderBook`` and a deliberately simple
reference implementation that just scans lists. The two must produce exactly
the same trades and the same resting book, and the optimised book must keep
its internal invariants after every step.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from lob import Order, OrderBook, OrderType, Side, TimeInForce


class NaiveBook:
    """O(n) reference matcher: obviously correct, deliberately slow."""

    def __init__(self):
        self.resting = []  # list of [seq, id, side, price, remaining]
        self.seq = 0

    def submit(self, oid, side, qty, price, tif):
        self.seq += 1
        opp = [r for r in self.resting if r[2] != side]
        if price is None:
            ok = lambda r: True  # noqa: E731
        elif side is Side.BUY:
            ok = lambda r: r[3] <= price  # noqa: E731
        else:
            ok = lambda r: r[3] >= price  # noqa: E731
        sign = 1 if side is Side.BUY else -1
        queue = sorted((r for r in opp if ok(r)), key=lambda r: (sign * r[3], r[0]))
        if tif is TimeInForce.FOK and sum(r[4] for r in queue) < qty:
            return []
        trades = []
        for r in queue:
            if qty == 0:
                break
            q = min(qty, r[4])
            r[4] -= q
            qty -= q
            trades.append((r[3], q, r[1]))
        self.resting = [r for r in self.resting if r[4] > 0]
        if qty and price is not None and tif is TimeInForce.GTC:
            self.resting.append([self.seq, oid, side, price, qty])
        return trades

    def cancel(self, oid):
        self.resting = [r for r in self.resting if r[1] != oid]

    def levels(self, side):
        agg = {}
        for r in self.resting:
            if r[2] is side:
                agg[r[3]] = agg.get(r[3], 0) + r[4]
        return sorted(agg.items(), reverse=side is Side.BUY)


order_strategy = st.tuples(
    st.sampled_from(["limit", "market", "cancel"]),
    st.sampled_from([Side.BUY, Side.SELL]),
    st.integers(1, 20),  # quantity
    st.integers(95, 105),  # price
    st.sampled_from(list(TimeInForce)),
    st.integers(0, 200),  # cancel target
)


def check_invariants(book: OrderBook):
    if book.best_bid is not None and book.best_ask is not None:
        assert book.best_bid < book.best_ask, "book is crossed"
    count = 0
    for side in (Side.BUY, Side.SELL):
        for _, level in book._side(side).items():
            assert level.orders, "empty price level left behind"
            assert level.volume == sum(o.remaining for o in level.orders.values())
            count += len(level)
    assert count == len(book)


@settings(max_examples=300, deadline=None)
@given(st.lists(order_strategy, min_size=1, max_size=200))
def test_matches_reference_implementation(ops):
    fast, slow = OrderBook(), NaiveBook()
    for oid, (kind, side, qty, px, tif, target) in enumerate(ops, start=1):
        if kind == "cancel":
            fast.cancel(target)
            slow.cancel(target)
        else:
            price = px if kind == "limit" else None
            otype = OrderType.LIMIT if kind == "limit" else OrderType.MARKET
            rep = fast.submit(Order(oid, side, qty, price, order_type=otype, tif=tif))
            expected = slow.submit(oid, side, qty, price, tif)
            got = [
                (t.price, t.quantity, t.sell_order_id if side is Side.BUY else t.buy_order_id)
                for t in rep.trades
            ]
            assert got == expected
        check_invariants(fast)
        snap = fast.depth(100)
        assert snap.bids == slow.levels(Side.BUY)
        assert snap.asks == slow.levels(Side.SELL)
