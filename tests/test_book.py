import pytest

from lob import Order, OrderBook, OrderStatus, OrderType, Side, TimeInForce


def limit(oid, side, qty, px, **kw):
    return Order(oid, side, qty, px, **kw)


def market(oid, side, qty, **kw):
    return Order(oid, side, qty, order_type=OrderType.MARKET, **kw)


@pytest.fixture
def book():
    b = OrderBook("TEST")
    b.submit(limit(1, Side.SELL, 10, 101))
    b.submit(limit(2, Side.SELL, 5, 101))
    b.submit(limit(3, Side.SELL, 20, 103))
    b.submit(limit(4, Side.BUY, 10, 99))
    b.submit(limit(5, Side.BUY, 15, 98))
    return b


def test_top_of_book(book):
    assert book.best_bid == 99
    assert book.best_ask == 101
    assert book.spread == 2
    assert book.mid == 100
    assert book.volume_at(Side.SELL, 101) == 15
    assert len(book) == 5


def test_non_crossing_order_rests(book):
    rep = book.submit(limit(6, Side.BUY, 7, 100))
    assert rep.status is OrderStatus.NEW
    assert rep.trades == ()
    assert book.best_bid == 100


def test_time_priority_within_level(book):
    rep = book.submit(limit(6, Side.BUY, 12, 101))
    assert rep.status is OrderStatus.FILLED
    assert [(t.sell_order_id, t.quantity) for t in rep.trades] == [(1, 10), (2, 2)]
    assert book.get_order(2).remaining == 3
    assert 1 not in book


def test_price_priority_and_trade_price(book):
    rep = book.submit(limit(6, Side.BUY, 25, 103))
    assert [(t.price, t.quantity) for t in rep.trades] == [(101, 10), (101, 5), (103, 10)]
    assert rep.average_price == pytest.approx((101 * 15 + 103 * 10) / 25)
    assert book.best_ask == 103
    assert book.volume_at(Side.SELL, 103) == 10


def test_partial_fill_rests_remainder(book):
    rep = book.submit(limit(6, Side.BUY, 20, 101))
    assert rep.status is OrderStatus.PARTIALLY_FILLED
    assert rep.filled_quantity == 15
    assert book.best_bid == 101
    assert book.volume_at(Side.BUY, 101) == 5
    assert book.best_ask == 103


def test_market_order_sweeps_and_cancels_rest(book):
    rep = book.submit(market(6, Side.SELL, 30))
    assert [(t.price, t.quantity) for t in rep.trades] == [(99, 10), (98, 15)]
    assert rep.status is OrderStatus.CANCELLED
    assert rep.remaining == 5
    assert book.best_bid is None


def test_ioc_does_not_rest(book):
    rep = book.submit(limit(6, Side.BUY, 20, 101, tif=TimeInForce.IOC))
    assert rep.filled_quantity == 15
    assert rep.status is OrderStatus.CANCELLED
    assert 6 not in book


def test_fok_rejects_when_not_fully_fillable(book):
    rep = book.submit(limit(6, Side.BUY, 16, 101, tif=TimeInForce.FOK))
    assert rep.status is OrderStatus.REJECTED
    assert rep.trades == ()
    assert book.volume_at(Side.SELL, 101) == 15  # book untouched


def test_fok_fills_across_levels(book):
    rep = book.submit(limit(6, Side.BUY, 35, 103, tif=TimeInForce.FOK))
    assert rep.status is OrderStatus.FILLED
    assert rep.filled_quantity == 35


def test_cancel(book):
    assert book.cancel(1)
    assert not book.cancel(1)
    assert book.volume_at(Side.SELL, 101) == 5
    book.cancel(2)
    assert book.best_ask == 103


def test_amend_down_keeps_priority(book):
    book.amend(1, new_quantity=4)
    rep = book.submit(limit(6, Side.BUY, 4, 101))
    assert rep.trades[0].sell_order_id == 1


def test_amend_up_loses_priority(book):
    book.amend(1, new_quantity=12)
    rep = book.submit(limit(6, Side.BUY, 4, 101))
    assert rep.trades[0].sell_order_id == 2


def test_amend_price_can_trade(book):
    rep = book.amend(4, new_price=101)
    assert rep.filled_quantity == 10
    assert book.best_bid == 98


def test_duplicate_id_rejected(book):
    with pytest.raises(ValueError):
        book.submit(limit(1, Side.BUY, 1, 90))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(order_id=1, side=Side.BUY, quantity=0, price=100),
        dict(order_id=1, side=Side.BUY, quantity=1),
        dict(order_id=1, side=Side.BUY, quantity=1, price=100, order_type=OrderType.MARKET),
    ],
)
def test_invalid_orders(kwargs):
    with pytest.raises(ValueError):
        Order(**kwargs)


def test_depth_and_imbalance(book):
    snap = book.depth(2)
    assert snap.bids == [(99, 10), (98, 15)]
    assert snap.asks == [(101, 15), (103, 20)]
    assert book.imbalance(1) == pytest.approx((10 - 15) / 25)


def test_trade_owners():
    b = OrderBook()
    b.submit(limit(1, Side.SELL, 5, 100, owner="alice"))
    rep = b.submit(market(2, Side.BUY, 5, owner="bob"))
    t = rep.trades[0]
    assert (t.buyer, t.seller, t.aggressor_side) == ("bob", "alice", Side.BUY)
