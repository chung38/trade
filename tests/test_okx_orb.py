"""OKX 模擬盤當沖執行器:用假交易所驗證,不連網路。"""

from __future__ import annotations

import csv

import pytest

from core.live.okx_orb import (
    BAR_SECONDS, DAY_SECONDS, DayState, OrbRunner, decide, load_okx_broker_class,
)

DAY = 20000 * DAY_SECONDS          # 某個 UTC 日的 00:00


def bar(n, o, h, l, c):
    return [DAY + n * BAR_SECONDS, o, h, l, c]


OPENING = [bar(0, 100, 101, 99, 100), bar(1, 100, 102, 99.5, 101),
           bar(2, 101, 101.5, 98, 99), bar(3, 99, 100, 98.5, 100)]


def test_range_needs_four_bars_and_starts_at_midnight():
    st = DayState(day=DAY // DAY_SECONDS)
    assert decide(st, OPENING[:3], DAY + 3 * BAR_SECONDS).kind == "none"
    assert st.range_high is None
    decide(st, OPENING, DAY + 4 * BAR_SECONDS)
    assert (st.range_high, st.range_low) == (102, 98)


def test_buy_on_close_above_high_once_per_day():
    st = DayState(day=DAY // DAY_SECONDS)
    bars = OPENING + [bar(4, 100, 102.5, 100, 101.9)]
    assert decide(st, bars, DAY + 5 * BAR_SECONDS).kind == "none"   # 只碰到、沒收上去
    bars.append(bar(5, 101.9, 103, 101.5, 102.5))
    assert decide(st, bars, DAY + 6 * BAR_SECONDS).kind == "buy"
    st.traded = True
    bars.append(bar(6, 102.5, 104, 102, 103.5))
    assert decide(st, bars, DAY + 7 * BAR_SECONDS).kind == "none"


def test_stop_at_range_low_and_forced_close_at_end_of_day():
    st = DayState(day=DAY // DAY_SECONDS, range_high=102, range_low=98, traded=True,
                  qty=0.5, entry_price=102.5, last_bar_seen=DAY + 5 * BAR_SECONDS)
    bars = OPENING + [bar(5, 101, 103, 101, 102.5), bar(6, 102, 102, 97.9, 98.5)]
    assert decide(st, bars, DAY + 7 * BAR_SECONDS).kind == "sell"

    st2 = DayState(day=DAY // DAY_SECONDS, range_high=102, range_low=98,
                   traded=True, qty=0.5, entry_price=102.5)
    assert decide(st2, OPENING, DAY + DAY_SECONDS - BAR_SECONDS).kind == "sell"
    assert decide(st2, [], DAY + DAY_SECONDS + 60).kind == "sell"     # 跨日還有倉


def test_no_entry_in_last_bar():
    st = DayState(day=DAY // DAY_SECONDS)
    bars = OPENING + [bar(94, 100, 105, 100, 104)]
    assert decide(st, bars, DAY + DAY_SECONDS - BAR_SECONDS + 1).kind == "none"


class FakeExchange:
    def __init__(self):
        self.markets = {"BTC/USDT": {"symbol": "BTC/USDT", "spot": True, "active": True,
                                     "limits": {"amount": {"min": 0.00001},
                                                "cost": {"min": 1}}}}
        self.bars = []
        self.price = 100.0
        self.base_free = 0.0
        self.orders = []
        self.sandbox = None

    def set_sandbox_mode(self, enabled):
        self.sandbox = enabled

    def load_markets(self):
        return self.markets

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        assert timeframe == "15m"
        return [[b[0] * 1000, *b[1:], 1.0] for b in self.bars if b[0] * 1000 >= since]

    def fetch_ticker(self, symbol):
        return {"last": self.price}

    def fetch_balance(self):
        return {"USDT": {"free": 1000, "total": 1000}, "BTC": {"free": self.base_free},
                "total": {"USDT": 1000, "BTC": self.base_free}}

    def amount_to_precision(self, symbol, amount):
        return f"{int(amount * 1e6) / 1e6:.6f}"

    def create_order(self, symbol, type_, side, amount, price, params):
        self.orders.append((side, amount))
        if side == "buy":
            self.base_free += amount * 0.999      # 手續費從幣扣
        else:
            self.base_free -= amount
        return {"id": str(len(self.orders))}

    def fetch_order(self, order_id, symbol):
        side, amount = self.orders[int(order_id) - 1]
        fee = ({"cost": amount * 0.001, "currency": "BTC"} if side == "buy"
               else {"cost": amount * self.price * 0.001, "currency": "USDT"})
        return {"id": order_id, "filled": amount, "average": self.price,
                "status": "closed", "fee": fee}


def _runner(tmp_path, demo=True):
    OKXBroker = load_okx_broker_class()
    ex = FakeExchange()
    broker = OKXBroker("k", "s", "p", demo=demo, exchange=ex)
    broker.connect()
    broker.confirm_live_trading(i_understand_the_risk=True)
    runner = OrbRunner(broker, "BTC/USDT", 50, tmp_path / "state.json",
                       tmp_path / "trades.csv", log=lambda *_: None, sleep=lambda _: None)
    return runner, ex


def test_runner_refuses_real_money(tmp_path):
    with pytest.raises(SystemExit, match="模擬"):
        _runner(tmp_path, demo=False)


def test_full_day_cycle_persists_state_and_logs_trade(tmp_path):
    runner, ex = _runner(tmp_path)
    ex.base_free = 0.5          # 模擬盤原本就有的 BTC 不可被賣掉
    ex.bars = OPENING + [bar(4, 100, 103, 100, 102.5)]
    ex.price = 102.5
    assert runner.step(DAY + 5 * BAR_SECONDS + 5).kind == "buy"
    assert ex.orders[0][0] == "buy"
    assert ex.orders[0][1] == pytest.approx(50 / 102.5, abs=1e-6)

    # 重開程式:從檔案讀回狀態,不會再買一次
    runner2 = OrbRunner(runner.broker, "BTC/USDT", 50, tmp_path / "state.json",
                        tmp_path / "trades.csv", log=lambda *_: None, sleep=lambda _: None)
    assert runner2.state.qty > 0 and runner2.state.traded
    ex.bars.append(bar(5, 102.5, 104, 102, 103))
    assert runner2.step(DAY + 6 * BAR_SECONDS + 5).kind == "none"
    assert len(ex.orders) == 1

    # 收盤前強制平倉;賣出量不超過扣掉手續費後的可用餘額
    ex.price = 104.0
    assert runner2.step(DAY + DAY_SECONDS - BAR_SECONDS + 5).kind == "sell"
    side, amount = ex.orders[-1]
    assert side == "sell" and amount == pytest.approx(ex.orders[0][1] * 0.999, abs=2e-6)
    assert runner2.state.qty == 0
    rows = list(csv.DictReader((tmp_path / "trades.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["tag"] == "ORB:收盤前強制平倉"
    qty = ex.orders[0][1]
    expected = 104 * qty * 0.999 * (1 - 0.001) - 102.5 * qty
    assert float(rows[0]["pnl"]) == pytest.approx(expected, abs=1e-6)
    assert rows[0]["pnl_currency"] == "USDT"

    # 隔天換新的一天,可以再進場
    assert runner2.step(DAY + DAY_SECONDS + 60).kind == "none"
    assert runner2.state.day == DAY // DAY_SECONDS + 1
    assert runner2.state.traded is False


def test_failed_buy_does_not_retry_same_day(tmp_path):
    runner, ex = _runner(tmp_path)
    ex.bars = OPENING + [bar(4, 100, 103, 100, 102.5)]
    ex.price = 102.5

    def boom(*a, **k):
        raise RuntimeError("Insufficient balance")
    ex.create_order = boom
    runner.step(DAY + 5 * BAR_SECONDS + 5)
    assert runner.state.traded is True and runner.state.qty == 0
    ex.bars.append(bar(5, 102.5, 104, 102, 103.5))
    assert runner.step(DAY + 6 * BAR_SECONDS + 5).kind == "none"


def test_trade_log_is_readable_by_analyzer(tmp_path):
    from core.ingest.loader import load_trades

    runner, ex = _runner(tmp_path)
    for d in range(3):
        start = DAY + d * DAY_SECONDS
        ex.bars = [[b[0] + d * DAY_SECONDS, *b[1:]] for b in OPENING]
        ex.bars.append([start + 4 * BAR_SECONDS, 100, 103, 100, 102.5])
        ex.price = 102.5
        runner.step(start + 5 * BAR_SECONDS + 5)
        ex.price = 101.0 + d
        runner.step(start + DAY_SECONDS - BAR_SECONDS + 5)
    trades = load_trades(str(tmp_path / "trades.csv"))
    assert len(trades) == 3
    assert all(t.pnl is not None for t in trades)
