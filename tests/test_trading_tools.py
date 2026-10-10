"""券商層 / 圖表層 / 腳架產生器的單元測試。

執行: python tests/test_trading_tools.py
或    python -m pytest tests/test_trading_tools.py -v
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import sys
import tempfile
import types
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.broker import (  # noqa: E402
    AccountInfo,
    Order,
    OrderResult,
    OrderSide,
    OrderType,
    PaperBroker,
    Position,
    list_brokers,
)
from core.broker.base import BrokerAdapter  # noqa: E402
from core.charts import build_preview_page, get_chart_lib, list_chart_libs  # noqa: E402
from core.scaffold import ScaffoldOptions, generate_project  # noqa: E402
from core.scaffold.generator import write_project  # noqa: E402
from core.verdict.judge import judge  # noqa: E402
from core.models import Market, Side, Trade, TradeLog  # noqa: E402
from datetime import datetime  # noqa: E402


def _execute_generated_main(opts: ScaffoldOptions) -> dict:
    """執行產出的 main.py,回傳命名空間供 runtime 安全閘測試。"""
    main_src = next(
        f.content for f in generate_project(opts) if f.relpath == "main.py"
    )
    modules = {
        "strategy": types.ModuleType("strategy"),
        "broker_setup": types.ModuleType("broker_setup"),
        "data_feed": types.ModuleType("data_feed"),
        "charting": types.ModuleType("charting"),
        "broker_lib": types.ModuleType("broker_lib"),
    }
    modules["strategy"].Strategy = object
    modules["broker_setup"].build_broker = lambda config: None
    modules["data_feed"].load_history = lambda symbol: []
    modules["charting"].render = lambda *args, **kwargs: None
    modules["broker_lib"].Order = object
    modules["broker_lib"].OrderSide = object
    modules["broker_lib"].BrokerAdapter = object

    namespace = {"__name__": "generated_scaffold_main"}
    with patch.dict(sys.modules, modules):
        exec(compile(main_src, "generated/main.py", "exec"), namespace)
    return namespace


def _execute_generated_strategy(opts: ScaffoldOptions) -> dict:
    strategy_src = next(
        f.content for f in generate_project(opts) if f.relpath == "strategy.py"
    )
    module = types.ModuleType("generated_scaffold_strategy")
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(
            compile(strategy_src, "generated/strategy.py", "exec"),
            module.__dict__,
        )
    return module.__dict__


def _execute_broker_template(key: str) -> dict:
    """把 scaffold 內嵌 broker 範本當成獨立產物載入。"""
    from core.broker import BROKER_TEMPLATES

    broker_lib = types.ModuleType("broker_lib")
    for value in (
        AccountInfo, BrokerAdapter, Order, OrderResult,
        OrderSide, OrderType, Position,
    ):
        setattr(broker_lib, value.__name__, value)
    namespace = {"__name__": f"generated_{key}_broker"}
    with patch.dict(sys.modules, {"broker_lib": broker_lib}):
        exec(
            compile(BROKER_TEMPLATES[key].code, f"brokers/{key}_broker.py", "exec"),
            namespace,
        )
    return namespace


class _PionexResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


class _PionexSession:
    def __init__(self, response: _PionexResponse):
        self.response = response
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.response


class _LiveBrokerProbe:
    is_live = True

    def __init__(self) -> None:
        self.confirmed_with = None

    def confirm_live_trading(self, *, i_understand_the_risk: bool = False) -> None:
        self.confirmed_with = i_understand_the_risk


# ── PaperBroker ──────────────────────────────────────────────
def test_paper_broker_buy_sell_cycle():
    b = PaperBroker(cash=100_000)
    b.connect()
    b.set_price("AAPL", 200.0)
    r = b.place_order(Order("AAPL", OrderSide.BUY, 100))
    assert r.ok
    assert b.get_positions()[0].quantity == 100
    b.set_price("AAPL", 220.0)
    # 帳戶權益應反映漲價
    assert b.get_account().equity > 100_000
    r2 = b.place_order(Order("AAPL", OrderSide.SELL, 100))
    assert r2.ok
    # 平倉後賺錢(扣成本仍應 > 起始)
    assert b.get_account().cash > 100_000


def test_paper_broker_insufficient_funds():
    b = PaperBroker(cash=1000)
    b.set_price("AAPL", 200.0)
    r = b.place_order(Order("AAPL", OrderSide.BUY, 100))  # 需 20000,只有 1000
    assert not r.ok
    assert "資金不足" in r.message


def test_paper_broker_rejects_naked_short():
    """賣出超過持有部位(裸放空)應被拒單 —— 本紙上模擬只支援做多。"""
    b = PaperBroker(cash=1e9, fee_rate=0, slippage=0)
    b.set_price("AAPL", 100.0)
    b.place_order(Order("AAPL", OrderSide.BUY, 100))   # +100
    b.set_price("AAPL", 200.0)
    r = b.place_order(Order("AAPL", OrderSide.SELL, 150))  # 想賣 150,只有 100
    assert not r.ok
    assert "放空" in r.message or "超過" in r.message
    # 部位不變(拒單後仍持 100)
    assert b.get_positions()[0].quantity == 100


def test_paper_broker_short_does_not_inflate_cash():
    """無部位直接賣出(放空)不應讓現金憑空增加(回歸測試 #1)。"""
    b = PaperBroker(cash=100_000, fee_rate=0, slippage=0)
    b.set_price("AAPL", 100.0)
    r = b.place_order(Order("AAPL", OrderSide.SELL, 50))   # 空手放空
    assert not r.ok                                        # 應被拒
    assert b.get_account().cash == 100_000                 # 現金不變


def test_paper_broker_reverse_reduce_keeps_avg_price():
    """反向減碼但未超量時,均價應維持原值。"""
    b = PaperBroker(cash=1e9, fee_rate=0, slippage=0)
    b.set_price("AAPL", 100.0)
    b.place_order(Order("AAPL", OrderSide.BUY, 100))   # +100 @ 100
    b.set_price("AAPL", 200.0)
    b.place_order(Order("AAPL", OrderSide.SELL, 30))   # 賣 30 → 仍持 +70
    pos = b.get_positions()[0]
    assert pos.quantity == 70
    assert abs(pos.avg_price - 100.0) < 1e-6   # 均價不變


def test_paper_broker_requires_price():
    b = PaperBroker()
    try:
        b.get_price("UNKNOWN")
        assert False, "應該要拋出缺報價錯誤"
    except ValueError:
        pass


def test_order_validation():
    try:
        Order("X", OrderSide.BUY, 0).validate()
        assert False
    except ValueError:
        pass
    with pytest.raises(ValueError, match="OrderSide"):
        Order("X", "BUY", 1).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="OrderType"):
        Order("X", OrderSide.BUY, 1, "MARKET").validate()  # type: ignore[arg-type]
    try:
        Order("X", OrderSide.BUY, 1, OrderType.LIMIT).validate()  # 限價單缺 limit_price
        assert False
    except ValueError:
        pass


# ── 安全閘門 ─────────────────────────────────────────────────
class _FakeLive(BrokerAdapter):
    name = "fake"
    is_live = True

    def connect(self): pass
    def get_account(self): pass
    def get_positions(self): return []
    def get_price(self, s): return 1.0
    def place_order(self, o):
        self._guard_live()
        return OrderResult(ok=True)
    def cancel_order(self, i): return False


def test_live_guard_blocks_unconfirmed():
    b = _FakeLive()
    try:
        b.place_order(Order("X", OrderSide.BUY, 1))
        assert False, "未確認時真實下單應被擋下"
    except PermissionError:
        pass


def test_confirm_requires_explicit_flag():
    b = _FakeLive()
    try:
        b.confirm_live_trading()
        assert False
    except PermissionError:
        pass
    b.confirm_live_trading(i_understand_the_risk=True)
    # 確認後應放行
    assert b.place_order(Order("X", OrderSide.BUY, 1)).ok


def test_paper_broker_is_not_live():
    assert PaperBroker().is_live is False


# ── 圖表 ─────────────────────────────────────────────────────
def test_chart_libs_registered():
    keys = {lib.key for lib in list_chart_libs()}
    assert keys == {"lightweight", "plotly", "mplfinance", "echarts"}
    for lib in list_chart_libs():
        assert "def render(" in lib.module_code


def test_chart_preview_generates():
    with tempfile.TemporaryDirectory() as d:
        out = build_preview_page(str(Path(d) / "preview.html"))
        content = Path(out).read_text(encoding="utf-8")
        assert "lightweight" in content
        assert "echarts" in content
        assert len(content) > 5000


def test_brokers_registered():
    keys = {t.key for t in list_brokers()}
    # 至少要涵蓋這些(台股 / 美股 / 加密貨幣各有代表);未來可再擴充
    expected = {
        "binance", "ibkr", "alpaca", "shioaji",          # 原有
        "yuanta", "fubon", "kgi", "tw_futures",          # 台股新增
        "pionex", "ccxt", "okx", "bybit", "tradier",    # 其他市場新增
    }
    assert expected <= keys


def test_pionex_scaffold_uses_official_production_api_without_fake_sandbox():
    from core.broker import BROKER_TEMPLATES

    template = BROKER_TEMPLATES["pionex"]
    assert template.market == "crypto"
    assert "https://api.pionex.com" in template.code
    assert "https://www.pionex.com/docs/api-docs/zh-hant" in template.code
    assert "testnet" not in template.code.lower()
    assert "sandbox" in template.notes.lower()  # 必須明說官方未列 sandbox

    files = generate_project(ScaffoldOptions(
        project_name="pionex-safe",
        broker="pionex",
        market="crypto",
        symbols=("BTC_USDT",),
    ))
    by_path = {item.relpath: item.content for item in files}
    assert "brokers/pionex_broker.py" in by_path
    config = by_path["config.example.yaml"]
    assert 'api_key_env: "PIONEX_API_KEY"' in config
    assert 'api_secret_env: "PIONEX_API_SECRET"' in config
    assert 'quote_currency: "USDT"' in config
    assert 'api_key: ""' not in config
    assert 'api_secret: ""' not in config
    assert "sandbox:" not in by_path["config.example.yaml"]
    assert "requests>=2.31" in by_path["requirements.txt"]
    assert 'os.environ[creds.get("api_key_env"' in by_path["broker_setup.py"]


def test_pionex_signature_sorts_query_and_signs_exact_json_body():
    PionexBroker = _execute_broker_template("pionex")["PionexBroker"]
    broker = PionexBroker("public-key", "secret-key")
    response = _PionexResponse({"result": True, "data": {"orderId": 7}})
    session = _PionexSession(response)
    broker.session = session
    body = {"symbol": "BTC_USDT", "side": "SELL", "type": "MARKET", "size": "0.01"}

    broker._private_request(
        "POST",
        "/api/v1/trade/order",
        params={"timestamp": 1655896754515, "z": "last", "a": "first"},
        body=body,
    )

    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "https://api.pionex.com/api/v1/trade/order"
    assert kwargs["params"] == [
        ("a", "first"), ("timestamp", 1655896754515), ("z", "last"),
    ]
    body_text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    assert kwargs["data"] == body_text
    signed = (
        "POST/api/v1/trade/order?"
        "a=first&timestamp=1655896754515&z=last" + body_text
    )
    expected = hmac.new(
        b"secret-key", signed.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    assert kwargs["headers"]["PIONEX-SIGNATURE"] == expected
    assert kwargs["headers"]["PIONEX-KEY"] == "public-key"
    assert kwargs["allow_redirects"] is False

    cancel_body = {"symbol": "BTC_USDT", "orderId": 7}
    broker._private_request(
        "DELETE",
        "/api/v1/trade/order",
        params={"timestamp": 1655896754515},
        body=cancel_body,
    )
    method, _, kwargs = session.calls[1]
    cancel_text = json.dumps(cancel_body, ensure_ascii=False, separators=(",", ":"))
    cancel_payload = (
        "DELETE/api/v1/trade/order?timestamp=1655896754515" + cancel_text
    )
    assert method == "DELETE"
    assert kwargs["data"] == cancel_text
    assert kwargs["headers"]["PIONEX-SIGNATURE"] == hmac.new(
        b"secret-key", cancel_payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    template_time = broker._private_request.__globals__["time"]
    with patch.object(template_time, "time", return_value=1655896754.515):
        broker._private_request(
            "GET", "/api/v1/trade/order", params={"orderId": 7},
        )
    method, _, kwargs = session.calls[2]
    assert method == "GET"
    assert kwargs["params"] == [("orderId", 7), ("timestamp", 1655896754515)]
    assert kwargs["data"] is None


def test_pionex_result_false_and_rate_limit_fail_closed():
    PionexBroker = _execute_broker_template("pionex")["PionexBroker"]
    broker = PionexBroker("key", "secret")

    for timeout in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(ValueError, match="timeout"):
            PionexBroker("key", "secret", timeout=timeout)

    bad = _PionexResponse({
        "result": False,
        "code": "INVALID_TIMESTAMP",
        "message": "Invalid timestamp",
    }, status_code=401)
    with pytest.raises(RuntimeError, match="同步系統時間"):
        broker._decode_response(bad)

    limited = _PionexResponse(
        {"result": False, "code": "RATE_LIMIT", "message": "too fast"},
        status_code=429,
    )
    with pytest.raises(RuntimeError, match="至少 60 秒"):
        broker._decode_response(limited)
    with pytest.raises(RuntimeError, match="冷卻中"):
        broker._rate_gate(1, private=False)

    class NonJson429:
        status_code = 429
        ok = False

        def json(self):
            raise ValueError("not json")

    with pytest.raises(RuntimeError, match="至少 60 秒"):
        PionexBroker("key", "secret")._decode_response(NonJson429())

    redirect = _PionexResponse({"result": True, "data": {}}, status_code=302)
    with pytest.raises(RuntimeError, match="HTTP 302"):
        PionexBroker("key", "secret")._decode_response(redirect)

    malformed = _PionexResponse({
        "result": False, "code": None, "message": 123,
    })
    with pytest.raises(RuntimeError, match="UNKNOWN: 123"):
        PionexBroker("key", "secret")._decode_response(malformed)


def test_pionex_market_buy_and_client_order_id_are_safe():
    PionexBroker = _execute_broker_template("pionex")["PionexBroker"]
    broker = PionexBroker("key", "secret")
    market_buy = Order("BTC_USDT", OrderSide.BUY, 0.001)

    with pytest.raises(PermissionError):
        broker.place_order(market_buy)
    broker.confirm_live_trading(i_understand_the_risk=True)
    result = broker.place_order(market_buy)
    assert not result.ok
    assert "amount" in result.message
    assert "quantity" in result.message

    # 中文且可重複的策略 client_tag 不可被誤用成 Pionex clientOrderId。
    limit_order = Order(
        "BTC_USDT", OrderSide.SELL, 0.001,
        OrderType.LIMIT, limit_price=100000, client_tag="觸發停利",
    )
    first_id = broker._new_client_order_id()
    second_id = broker._new_client_order_id()
    assert re.fullmatch(r"[A-Za-z0-9-]{1,64}", first_id)
    assert first_id != second_id
    assert limit_order.client_tag == "觸發停利"

    invalid_id = Order(
        "BTC_USDT", OrderSide.SELL, 0.001,
        OrderType.LIMIT, limit_price=100000,
        client_tag="觸發停利", client_order_id="中文 ID",
    )
    result = broker.place_order(invalid_id)
    assert not result.ok
    assert "client_order_id" in result.message

    invalid_type = Order(
        "BTC_USDT", OrderSide.SELL, 0.001,
        OrderType.LIMIT, limit_price=100000, client_order_id=123,
    )
    result = broker.place_order(invalid_type)
    assert not result.ok
    assert "必須是字串" in result.message

    broker._symbol_info = lambda pair: {
        "symbol": pair, "type": "SPOT", "enable": True,
        "baseCurrency": "BTC", "quoteCurrency": "USDT",
        "basePrecision": 8, "quotePrecision": 2,
        "minTradeSize": "0.0001", "maxTradeSize": "100",
        "minTradeDumping": "0.0001", "maxTradeDumping": "100",
        "minAmount": "10",
    }
    ambiguous = Order(
        "BTC_USDT", OrderSide.SELL, 0.001,
        OrderType.LIMIT, limit_price=100000,
        client_tag="觸發停利", client_order_id="safe-id-1",
    )
    for bad_data in ({}, {"orderId": None}, {"orderId": 0}, {"orderId": "abc"}):
        broker._private_request = lambda *args, _data=bad_data, **kwargs: {
            "result": True, "data": _data,
        }
        result = broker.place_order(ambiguous)
        assert not result.ok
        assert "結果不明" in result.message
        assert "safe-id-1" in result.message


def test_pionex_symbol_and_cancel_require_unambiguous_context():
    PionexBroker = _execute_broker_template("pionex")["PionexBroker"]
    broker = PionexBroker("key", "secret")
    assert broker._normalise_symbol("btc/usdt") == "BTC_USDT"
    assert broker._normalise_symbol("btc-usdt") == "BTC_USDT"
    with pytest.raises(ValueError, match="BTCUSDT"):
        broker._normalise_symbol("BTCUSDT")
    with pytest.raises(RuntimeError, match="basePrecision"):
        broker._validate_order_rules(
            {}, quantity=Decimal("1"), price=Decimal("1"),
            order_type="LIMIT", side="SELL", pair="BTC_USDT",
        )
    with pytest.raises(RuntimeError, match="basePrecision"):
        broker._validate_order_rules(
            {
                "basePrecision": 8.5, "quotePrecision": 2,
                "minTradeSize": "0.0001", "maxTradeSize": "100",
                "minAmount": "10",
            },
            quantity=Decimal("1"), price=Decimal("100"),
            order_type="LIMIT", side="SELL", pair="BTC_USDT",
        )
    with pytest.raises(PermissionError):
        broker.cancel_order("123")


def test_pionex_cancel_recovers_symbol_and_surfaces_partial_fill():
    PionexBroker = _execute_broker_template("pionex")["PionexBroker"]
    broker = PionexBroker("key", "secret")
    broker.confirm_live_trading(i_understand_the_risk=True)
    calls = []

    def fake_private(method, path, *, params=None, body=None, weight=1):
        calls.append((method, path, params, body, weight))
        if method == "DELETE":
            return {"result": True, "timestamp": 1}
        if path.endswith("fillsByOrderId"):
            return {"result": True, "data": {"fills": [
                {"orderId": 123, "price": "100000", "size": "0.2"}
            ]}}
        return {"result": True, "data": {
            "orderId": 123, "symbol": "BTC_USDT", "status": "CLOSED",
            "filledSize": "0.2", "filledAmount": "20000",
        }}

    broker._private_request = fake_private
    assert broker.cancel_order("123") is True
    assert [call[0:2] for call in calls] == [
        ("GET", "/api/v1/trade/order"),
        ("DELETE", "/api/v1/trade/order"),
        ("GET", "/api/v1/trade/order"),
        ("GET", "/api/v1/trade/fillsByOrderId"),
    ]
    assert broker.last_cancel_details["filled_size"] == "0.2"
    assert broker.last_cancel_details["fills"]
    assert "不代表零成交" in broker.last_cancel_details["warning"]
    with pytest.raises(ValueError, match="正整數"):
        broker.cancel_order("0")


def test_generated_crypto_sizing_and_unknown_cost_basis_are_fail_closed():
    opts = ScaffoldOptions(
        project_name="pionex-runtime-safety",
        broker="pionex",
        market="crypto",
        symbols=["BTC_USDT"],
    )
    main_ns = _execute_generated_main(opts)
    calculate = main_ns["calculate_order_quantity"]
    assert calculate(500, 100_000, "crypto") == pytest.approx(0.005)
    assert calculate(50, 100, "us_stock") == 0
    assert calculate(0, 100, "crypto") == 0

    broker_ns = _execute_broker_template("pionex")
    broker = broker_ns["PionexBroker"]("key", "secret")
    broker._balance_rows = lambda: [
        {"coin": "BTC", "free": "0.2", "frozen": "0"},
        {"coin": "USDT", "free": "100", "frozen": "0"},
    ]
    broker._spot_prices = lambda: {"BTC_USDT": Decimal("100000")}
    position = broker.get_positions()[0]
    assert position.cost_basis_known is False
    with pytest.raises(ValueError, match="成本基準"):
        _ = position.unrealized_pnl

    strategy_ns = _execute_generated_strategy(opts)
    strategy = strategy_ns["Strategy"]({"risk": {}})
    signal = strategy.on_bar(
        "BTC_USDT",
        [{"close": 100.0}],  # 與真實 BTC 市價差很多，也不可觸發出售
        position,
    )
    assert signal.action == "hold"
    assert "成本基準未知" in signal.reason


def test_all_broker_templates_valid_python():
    """每個券商範本都必須是合法 Python,且能抓出 Broker 類別名。"""
    import ast
    from core.broker import BROKER_TEMPLATES
    for key, t in BROKER_TEMPLATES.items():
        ast.parse(t.code)                       # 語法正確(會 raise 若錯)
        assert t.class_name.endswith("Broker"), f"{key} class_name 異常"
        assert "from broker_lib import" in t.code  # 自包含 import


def test_scaffold_new_broker_no_keyerror():
    """新增券商(如元大)產生專案時不應 KeyError(回歸測試)。"""
    opts = ScaffoldOptions(project_name="t", broker="yuanta", market="tw_stock")
    rels = {f.relpath for f in generate_project(opts)}
    assert "brokers/yuanta_broker.py" in rels


# ── 腳架產生器 ───────────────────────────────────────────────
def test_scaffold_generates_all_files():
    opts = ScaffoldOptions(project_name="t", broker="binance", chart="plotly")
    files = generate_project(opts)
    rels = {f.relpath for f in files}
    assert "main.py" in rels
    assert "strategy.py" in rels
    assert "charting.py" in rels
    assert "brokers/binance_broker.py" in rels
    assert "README.md" in rels


def test_scaffold_paper_has_no_broker_template_file():
    opts = ScaffoldOptions(project_name="t", broker="paper")
    rels = {f.relpath for f in generate_project(opts)}
    # paper 模式不應產生真實券商範例檔
    assert not any(r.startswith("brokers/") for r in rels)


def test_scaffold_discouraged_disables_live():
    # 用負期望交易紀錄產生裁決
    trades = [
        Trade("X", Market.US_STOCK, Side.LONG, datetime(2024, 1, 1),
              datetime(2024, 1, 2), 100, 100, 10, pnl=(-50 if i % 4 else 20))
        for i in range(40)
    ]
    verdict = judge(TradeLog(trades), n_bootstrap=500)
    assert verdict.should_discourage

    opts = ScaffoldOptions(project_name="t", verdict=verdict)
    config = next(f for f in generate_project(opts) if f.relpath == "config.example.yaml")
    assert "allow_live_trading: false" in config.content


def test_scaffold_live_flag_requires_complete_tiny_live_stage():
    from core.onboarding import StageAssessment

    paper_stage = StageAssessment(
        code="paper_until_oos",
        title="仍只適合紙上模擬",
        reason="樣本外尚未確認",
        next_actions=("累積新資料",),
        exit_code=2,
    )
    tiny_stage = StageAssessment(
        code="tiny_live_validation",
        title="只可極小額驗證",
        reason="樣本外與風險資料通過",
        next_actions=("限制風險",),
        exit_code=0,
    )

    for opts in (
        ScaffoldOptions(project_name="no-analysis"),
        ScaffoldOptions(project_name="paper-stage", stage=paper_stage),
    ):
        config = next(
            f for f in generate_project(opts)
            if f.relpath == "config.example.yaml"
        )
        assert "allow_live_trading: false" in config.content
        assert "stage_code:" in config.content

    paper_config = next(
        f for f in generate_project(
            ScaffoldOptions(project_name="paper-reason", stage=paper_stage)
        )
        if f.relpath == "config.example.yaml"
    )
    assert "樣本外尚未確認" in paper_config.content

    config = next(
        f for f in generate_project(
            ScaffoldOptions(project_name="tiny-stage", stage=tiny_stage)
        )
        if f.relpath == "config.example.yaml"
    )
    assert "allow_live_trading: true" in config.content


def test_generated_live_runtime_requires_all_stage_safety_gates():
    """產出的 runtime 必須實際攔截缺欄位、錯型別與非允許階段。"""
    namespace = _execute_generated_main(ScaffoldOptions(project_name="runtime-gate"))
    maybe_enable_live = namespace["maybe_enable_live"]
    valid = {
        "risk": {"i_have_read_disclaimer": True},
        "anti_gambling": {
            "allow_live_trading": True,
            "stage_code": "tiny_live_validation",
        },
    }

    # 舊總開關仍是獨立閘門；其餘設定全正確也不能繞過。
    broker = _LiveBrokerProbe()
    with pytest.raises(SystemExit, match="ALLOW_LIVE_TRADING"):
        maybe_enable_live(broker, valid)
    assert broker.confirmed_with is None

    namespace["ALLOW_LIVE_TRADING"] = True
    blocked_configs = [
        None,
        {
            "risk": {"i_have_read_disclaimer": False},
            "anti_gambling": valid["anti_gambling"],
        },
        {"risk": {"i_have_read_disclaimer": True}},
        {
            "risk": {"i_have_read_disclaimer": True},
            "anti_gambling": "not-a-mapping",
        },
        {
            "risk": {"i_have_read_disclaimer": True},
            "anti_gambling": {
                "allow_live_trading": False,
                "stage_code": "tiny_live_validation",
            },
        },
        {
            "risk": {"i_have_read_disclaimer": True},
            "anti_gambling": {
                "allow_live_trading": "true",
                "stage_code": "tiny_live_validation",
            },
        },
        {
            "risk": {"i_have_read_disclaimer": True},
            "anti_gambling": {"allow_live_trading": True},
        },
        {
            "risk": {"i_have_read_disclaimer": True},
            "anti_gambling": {
                "allow_live_trading": True,
                "stage_code": "paper_until_oos",
            },
        },
    ]
    for config in blocked_configs:
        broker = _LiveBrokerProbe()
        with pytest.raises(SystemExit):
            maybe_enable_live(broker, config)
        assert broker.confirmed_with is None

    broker = _LiveBrokerProbe()
    maybe_enable_live(broker, valid)
    assert broker.confirmed_with is True

    reject_replay = namespace["reject_live_historical_replay"]
    with pytest.raises(SystemExit, match="歷史／示範 K 線"):
        reject_replay(_LiveBrokerProbe())
    reject_replay(types.SimpleNamespace(is_live=False))

    # 完整 run() 在 live broker 進入 connect() 前就必須退出，送單路徑不可觸發。
    class LiveRunProbe(_LiveBrokerProbe):
        def __init__(self):
            super().__init__()
            self.connect_calls = 0

        def connect(self):
            self.connect_calls += 1

    live_probe = LiveRunProbe()
    namespace["load_config"] = lambda: valid
    namespace["build_broker"] = lambda config: live_probe
    with pytest.raises(SystemExit, match="歷史／示範 K 線"):
        namespace["run"]()
    assert live_probe.confirmed_with is True
    assert live_probe.connect_calls == 0


def test_generated_live_runtime_yaml_parse_failure_is_fail_closed(tmp_path):
    """破損 YAML 要退回禁止 live 的安全預設,不能留下半解析設定。"""
    namespace = _execute_generated_main(ScaffoldOptions(project_name="parse-gate"))
    config_path = tmp_path / "broken-config.yaml"
    config_path.write_text(
        "anti_gambling: [this is not valid YAML",
        encoding="utf-8",
    )

    config = namespace["load_config"](str(config_path))
    assert config["anti_gambling"] == {
        "stage_code": "unverified",
        "allow_live_trading": False,
    }

    namespace["ALLOW_LIVE_TRADING"] = True
    broker = _LiveBrokerProbe()
    with pytest.raises(SystemExit):
        namespace["maybe_enable_live"](broker, config)
    assert broker.confirmed_with is None


def test_scaffold_writes_runnable_project():
    """產出的專案應能寫入磁碟,且 main.py 含安全開關。"""
    with tempfile.TemporaryDirectory() as d:
        opts = ScaffoldOptions(project_name="bot", broker="paper", chart="lightweight")
        root = write_project(opts, d)
        assert (root / "main.py").exists()
        main_src = (root / "main.py").read_text(encoding="utf-8")
        assert "ALLOW_LIVE_TRADING = False" in main_src
        # 圖表模組應含 render
        assert "def render(" in (root / "charting.py").read_text(encoding="utf-8")


if __name__ == "__main__":
    import traceback

    _this = sys.modules[__name__]
    fns = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
        and getattr(v, "__module__", None) == _this.__name__
    ]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
