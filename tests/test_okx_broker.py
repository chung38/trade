"""OKX 現貨連接器範本:用假的 ccxt 物件驗證,不連網路。"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from core.broker.base import (
    AccountInfo, BrokerAdapter, Order, OrderResult, OrderSide, OrderType, Position,
)


def _okx_class():
    from core.broker import BROKER_TEMPLATES

    broker_lib = types.ModuleType("broker_lib")
    for value in (AccountInfo, BrokerAdapter, Order, OrderResult,
                  OrderSide, OrderType, Position):
        setattr(broker_lib, value.__name__, value)
    namespace = {"__name__": "generated_okx_broker"}
    with patch.dict(sys.modules, {"broker_lib": broker_lib}):
        exec(compile(BROKER_TEMPLATES["okx"].code, "brokers/okx_broker.py", "exec"),
             namespace)
    return namespace["OKXBroker"]


def _market(symbol, *, spot=True, active=True, min_amount=0.00001, min_cost=1.0):
    return {
        "symbol": symbol, "spot": spot, "active": active,
        "limits": {"amount": {"min": min_amount}, "cost": {"min": min_cost}},
    }


class FakeOKX:
    def __init__(self):
        self.sandbox = None
        self.markets = {}
        self.calls = []
        self.prices = {"BTC/USDT": 60000.0, "ETH/USDT": 3000.0}
        self.balance = {
            "USDT": {"free": 500.0, "total": 600.0},
            "total": {"USDT": 600.0, "BTC": 0.01, "ETH": 0.000001, "XYZ": 5.0},
        }
        self.create_error = None

    def set_sandbox_mode(self, enabled):
        self.sandbox = enabled

    def load_markets(self):
        self.markets = {
            "BTC/USDT": _market("BTC/USDT"),
            "ETH/USDT": _market("ETH/USDT", min_amount=0.0001),
            "BTC/USDT:USDT": _market("BTC/USDT:USDT", spot=False),
            "OLD/USDT": _market("OLD/USDT", active=False),
        }
        return self.markets

    def fetch_ticker(self, symbol):
        return {"last": self.prices[symbol]}

    def fetch_balance(self):
        return self.balance

    def amount_to_precision(self, symbol, amount):
        return f"{int(amount * 1e6) / 1e6:.6f}"

    def price_to_precision(self, symbol, price):
        return f"{price:.1f}"

    def create_order(self, symbol, type_, side, amount, price, params):
        self.calls.append(("create", symbol, type_, side, amount, price, dict(params)))
        if self.create_error:
            raise self.create_error
        return {"id": "123"}

    def fetch_order(self, order_id, symbol):
        self.calls.append(("fetch_order", order_id, symbol))
        return {"id": order_id, "filled": 0.001, "average": 60010.0, "status": "closed"}

    def cancel_order(self, order_id, symbol):
        self.calls.append(("cancel", order_id, symbol))
        return {"id": order_id}

    def fetch_open_orders(self, symbol):
        return [{"id": "9", "symbol": symbol}]


def _broker(**kw):
    OKXBroker = _okx_class()
    fake = FakeOKX()
    broker = OKXBroker("k", "s", "p", exchange=fake, **kw)
    broker.connect()
    return broker, fake


def test_demo_is_default_and_sets_sandbox():
    broker, fake = _broker()
    assert broker.demo is True
    assert fake.sandbox is True
    assert broker.is_live is True


def test_real_mode_does_not_enable_sandbox_and_rejects_non_bool_demo():
    broker, fake = _broker(demo=False)
    assert fake.sandbox is None
    with pytest.raises(ValueError, match="布林"):
        _okx_class()("k", "s", "p", demo="false")


def test_symbol_normalization_rejects_contracts_and_inactive():
    broker, _ = _broker()
    assert broker.normalize_symbol("btc-usdt") == "BTC/USDT"
    assert broker.normalize_symbol("BTC_USDT") == "BTC/USDT"
    assert broker.normalize_symbol("BTC") == "BTC/USDT"
    with pytest.raises(ValueError, match="現貨"):
        broker.normalize_symbol("BTC/USDT:USDT")
    with pytest.raises(ValueError, match="暫停"):
        broker.normalize_symbol("OLD/USDT")
    with pytest.raises(ValueError, match="沒有"):
        broker.normalize_symbol("NOPE/USDT")


def test_requires_connect_before_use():
    OKXBroker = _okx_class()
    broker = OKXBroker("k", "s", "p", exchange=FakeOKX())
    with pytest.raises(RuntimeError, match="connect"):
        broker.get_price("BTC/USDT")


def test_positions_skip_dust_and_unpriced_assets_and_mark_unknown_cost():
    broker, _ = _broker()
    positions = broker.get_positions()
    assert [p.symbol for p in positions] == ["BTC/USDT"]
    assert positions[0].quantity == pytest.approx(0.01)
    assert positions[0].market_price == 60000.0
    assert positions[0].cost_basis_known is False
    acct = broker.get_account()
    assert acct.cash == 500.0
    assert acct.equity == pytest.approx(600.0 + 0.01 * 60000.0)
    assert acct.currency == "USDT"


def test_place_order_blocked_until_confirmed():
    broker, fake = _broker()
    with pytest.raises(PermissionError):
        broker.place_order(Order("BTC/USDT", OrderSide.BUY, 0.001))
    with pytest.raises(PermissionError):
        broker.cancel_order("123", "BTC/USDT")
    assert fake.calls == []


def test_market_buy_sends_base_quantity_spot_cash_and_reports_fill():
    broker, fake = _broker()
    broker.confirm_live_trading(i_understand_the_risk=True)
    result = broker.place_order(Order("BTC-USDT", OrderSide.BUY, 0.0012345))
    assert result.ok and result.order_id == "123"
    assert result.filled_quantity == pytest.approx(0.001)
    assert result.avg_price == 60010.0
    _, symbol, type_, side, amount, price, params = fake.calls[0]
    assert (symbol, type_, side, price) == ("BTC/USDT", "market", "buy", None)
    assert amount == pytest.approx(0.001234)
    assert params["tdMode"] == "cash"
    assert params["tgtCcy"] == "base_ccy"
    assert len(params["clientOrderId"]) == 32 and params["clientOrderId"].isalnum()


def test_limit_sell_and_cancel_remembers_symbol():
    broker, fake = _broker()
    broker.confirm_live_trading(i_understand_the_risk=True)
    result = broker.place_order(Order("BTC/USDT", OrderSide.SELL, 0.001,
                                      order_type=OrderType.LIMIT, limit_price=70000.04,
                                      client_order_id="abc123"))
    assert result.ok
    create = fake.calls[0]
    assert create[2:6] == ("limit", "sell", pytest.approx(0.001), 70000.0)
    assert "tgtCcy" not in create[6]
    assert create[6]["clientOrderId"] == "abc123"
    assert broker.cancel_order("123") is True
    assert fake.calls[-1] == ("cancel", "123", "BTC/USDT")


def test_cancel_unknown_order_requires_symbol():
    broker, _ = _broker()
    broker.confirm_live_trading(i_understand_the_risk=True)
    with pytest.raises(ValueError, match="symbol"):
        broker.cancel_order("999")


def test_below_exchange_minimum_is_rejected_without_sending():
    broker, fake = _broker()
    broker.confirm_live_trading(i_understand_the_risk=True)
    tiny = broker.place_order(Order("BTC/USDT", OrderSide.BUY, 0.000001))
    assert not tiny.ok and "最小下單量" in tiny.message
    cheap = broker.place_order(Order("BTC/USDT", OrderSide.BUY, 0.00001))
    assert not cheap.ok and "最小下單金額" in cheap.message
    bad_id = broker.place_order(Order("BTC/USDT", OrderSide.BUY, 0.001,
                                      client_order_id="has-dash"))
    assert not bad_id.ok and "英數" in bad_id.message
    assert [c for c in fake.calls if c[0] == "create"] == []


def test_exchange_error_is_reported_not_raised():
    broker, fake = _broker()
    broker.confirm_live_trading(i_understand_the_risk=True)
    fake.create_error = RuntimeError("Insufficient balance")
    result = broker.place_order(Order("BTC/USDT", OrderSide.BUY, 0.001))
    assert not result.ok and "Insufficient balance" in result.message


def test_real_ccxt_request_carries_spot_cash_and_base_quantity():
    """若環境有 ccxt,確認參數真的變成 OKX 請求欄位(不連網)。"""
    ccxt = pytest.importorskip("ccxt")
    ex = ccxt.okx()
    ex.markets = None
    ex.set_markets([{
        "id": "BTC-USDT", "symbol": "BTC/USDT", "base": "BTC", "quote": "USDT",
        "baseId": "BTC", "quoteId": "USDT", "type": "spot", "spot": True,
        "margin": False, "swap": False, "future": False, "option": False,
        "contract": False, "active": True,
        "precision": {"amount": 0.00000001, "price": 0.1},
        "limits": {"amount": {"min": 0.00001}, "cost": {"min": 1}},
    }])
    request = ex.create_order_request(
        "BTC/USDT", "market", "buy", 0.001, None,
        {"tdMode": "cash", "tgtCcy": "base_ccy", "clientOrderId": "abc"},
    )
    assert request["instId"] == "BTC-USDT"
    assert request["tdMode"] == "cash"
    assert request["tgtCcy"] == "base_ccy"
    assert request["sz"] == "0.001"
    assert request["clOrdId"] == "abc"
    ex.set_sandbox_mode(True)
    assert ex.headers.get("x-simulated-trading") == "1"


def test_scaffold_okx_project_reads_keys_from_env_and_defaults_demo():
    from core.scaffold.generator import ScaffoldOptions, generate_project

    files = {f.relpath: f.content for f in generate_project(ScaffoldOptions(
        project_name="okx-bot", broker="okx", market="crypto", symbols=["BTC/USDT"]))}
    assert "brokers/okx_broker.py" in files
    setup = next(v for k, v in files.items() if k.endswith("broker_setup.py"))
    assert "OKX_API_PASSPHRASE" in setup and "OKXBroker" in setup
    config = next(v for k, v in files.items() if k.startswith("config"))
    assert "demo: true" in config and "passphrase_env" in config
