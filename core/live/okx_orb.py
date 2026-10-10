"""OKX 模擬盤當沖執行器:開盤 1 小時區間突破(Opening Range Breakout)。

規則(以 UTC 日為一天,也就是台灣時間每天 08:00 換日):
1. 開盤區間 = 當天前 4 根 15 分 K(UTC 00:00–01:00,台灣 08:00–09:00)的最高價與最低價。
2. 之後任何一根 15 分 K 收盤高於區間高點,就用固定金額市價買進。每天最多進場一次。
3. 價格跌到區間低點就停損賣出。
4. 當天最後一根 K(UTC 23:45 起,台灣隔天 07:45)一定全部賣出,不留倉過夜。

安全設計:
- 只允許 OKX 模擬交易(demo=True),程式裡沒有切到真錢的選項。
- 狀態寫在 JSON 檔,重開程式不會重複進場或忘記持倉。
- 每筆交易寫進 CSV,方便之後用反詐投資王分析。

回測(BTC 2018–2026,見專案說明)顯示這套規則扣掉手續費後是虧錢的;
這支程式的目的是用假錢親眼驗證,不是賺錢保證。

用法:
    pip install -e ".[okx]"
    set OKX_API_KEY=... / OKX_API_SECRET=... / OKX_API_PASSPHRASE=...(模擬盤 key)
    python -m core.live.okx_orb --symbol BTC/USDT --budget 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..broker import base as _base
from ..broker.base import Order, OrderSide

BAR_SECONDS = 15 * 60
DAY_SECONDS = 24 * 60 * 60
RANGE_BARS = 4                      # 前 4 根 15 分 K = 開盤 1 小時
LAST_BAR_START = DAY_SECONDS - BAR_SECONDS   # UTC 23:45 起強制平倉


def load_okx_broker_class():
    """從券商範本載入 OKXBroker,與 scaffold 產出的 brokers/okx_broker.py 同一份程式碼。"""
    from ..broker import BROKER_TEMPLATES

    broker_lib = types.ModuleType("broker_lib")
    for name in ("AccountInfo", "BrokerAdapter", "Order", "OrderResult",
                 "OrderSide", "OrderType", "Position"):
        setattr(broker_lib, name, getattr(_base, name))
    namespace = {"__name__": "okx_broker"}
    saved = sys.modules.get("broker_lib")
    sys.modules["broker_lib"] = broker_lib
    try:
        exec(compile(BROKER_TEMPLATES["okx"].code, "okx_broker.py", "exec"), namespace)
    finally:
        if saved is None:
            sys.modules.pop("broker_lib", None)
        else:
            sys.modules["broker_lib"] = saved
    return namespace["OKXBroker"]


@dataclass
class DayState:
    day: int = -1                    # UTC 日序號(epoch 秒 // 86400)
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    traded: bool = False             # 今天是否已進場過
    qty: float = 0.0                 # 目前持有數量(0 = 空手)
    entry_price: float = 0.0
    entry_cost: float = 0.0          # 實際付出的報價幣金額
    entry_time: int = 0
    entry_order_id: str = ""
    last_bar_seen: int = 0           # 已處理過的最後一根 K 的開始時間

    @classmethod
    def load(cls, path: Path) -> "DayState":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)


@dataclass
class Action:
    kind: str                        # "buy" | "sell" | "none"
    reason: str = ""


def decide(state: DayState, closed_bars: list, now: int) -> Action:
    """純函式:依今天已收完的 15 分 K 與目前狀態決定動作。

    closed_bars: [[open_time_sec, open, high, low, close], ...],只含今天、已收完的 K。
    會更新 state 的開盤區間,但不會改持倉(持倉由實際成交結果更新)。
    """
    today = now // DAY_SECONDS
    day_start = today * DAY_SECONDS
    seconds_into_day = now - day_start

    if state.qty > 0 and seconds_into_day >= LAST_BAR_START:
        return Action("sell", "收盤前強制平倉")
    if state.qty > 0 and state.day != today:
        return Action("sell", "跨日仍有持倉,立即平倉")

    bars = [b for b in closed_bars if day_start <= b[0] < day_start + DAY_SECONDS]
    if len(bars) >= RANGE_BARS and state.range_high is None:
        opening = bars[:RANGE_BARS]
        if opening[0][0] == day_start:
            state.range_high = max(b[2] for b in opening)
            state.range_low = min(b[3] for b in opening)
    if state.range_high is None:
        return Action("none", "開盤區間尚未形成")

    new_bars = [b for b in bars[RANGE_BARS:] if b[0] > state.last_bar_seen]
    if new_bars:
        state.last_bar_seen = new_bars[-1][0]

    if state.qty > 0:
        if any(b[3] <= state.range_low for b in new_bars):
            return Action("sell", f"跌破開盤區間低點 {state.range_low}")
        return Action("none", "持倉中")

    if state.traded or seconds_into_day >= LAST_BAR_START:
        return Action("none", "今天不再進場")
    if new_bars and new_bars[-1][4] > state.range_high:
        return Action("buy", f"收盤突破開盤區間高點 {state.range_high}")
    return Action("none", "等待突破")


def _iso(ts: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))


def _append_trade(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


class OrbRunner:
    def __init__(self, broker, symbol: str, budget: float, state_path: Path,
                 trades_path: Path, log=print, sleep=time.sleep):
        if getattr(broker, "demo", None) is not True:
            raise SystemExit("⛔ 這支執行器只允許 OKX 模擬交易(demo=True)。")
        if not (budget > 0):
            raise SystemExit("⛔ --budget 必須大於 0")
        self.broker = broker
        self.symbol = symbol
        self.budget = float(budget)
        self.state_path = state_path
        self.trades_path = trades_path
        self.log = log
        self.sleep = sleep
        self.state = DayState.load(state_path)

    def _today_bars(self, now: int) -> list:
        day_start = (now // DAY_SECONDS) * DAY_SECONDS
        raw = self.broker.exchange.fetch_ohlcv(
            self.symbol, "15m", since=day_start * 1000, limit=100)
        bars = []
        for t, o, h, l, c, *_ in raw:
            start = int(t) // 1000
            if start + BAR_SECONDS <= now:        # 只用已收完的 K
                bars.append([start, float(o), float(h), float(l), float(c)])
        return bars

    def _quote_fee(self, raw: dict, proceeds: float) -> float:
        """賣出手續費(報價幣)。交易所沒回報時,保守用 0.1% 估算。"""
        fee = (raw or {}).get("fee") or {}
        quote = self.symbol.split("/")[1]
        try:
            if str(fee.get("currency", "")).upper() == quote and fee.get("cost") is not None:
                return abs(float(fee["cost"]))
        except (TypeError, ValueError):
            pass
        return proceeds * 0.001

    def _base_fee(self, raw: dict) -> float:
        """買單從買到的幣扣的手續費(標的幣)。沒回報時用 0.1% 估算,寧可少賣不可多賣。"""
        fee = (raw or {}).get("fee") or {}
        base_ccy = self.symbol.split("/")[0]
        filled = float((raw or {}).get("filled") or 0)
        try:
            if str(fee.get("currency", "")).upper() == base_ccy and fee.get("cost") is not None:
                return abs(float(fee["cost"]))
            if str(fee.get("currency", "")).upper() and fee.get("cost") is not None:
                return 0.0           # 手續費用別的幣付(例如 OKB 抵扣),不影響數量
        except (TypeError, ValueError):
            pass
        return filled * 0.001

    def _free_base(self) -> float:
        base_ccy = self.symbol.split("/")[0]
        bal = self.broker.exchange.fetch_balance()
        return float((bal.get(base_ccy) or {}).get("free") or 0)

    def _wait_fill(self, result, tries: int = 3):
        """市價單偶爾在回應當下還沒成交完;再查幾次,避免持倉沒被記錄到。"""
        for _ in range(tries):
            self.sleep(1)
            try:
                detail = self.broker.exchange.fetch_order(result.order_id, self.symbol)
            except Exception:  # noqa: BLE001
                continue
            filled = float(detail.get("filled") or 0)
            if filled > 0:
                result.filled_quantity = filled
                result.avg_price = float(detail.get("average") or 0)
                return result
        return result

    def step(self, now: Optional[int] = None) -> Action:
        now = int(now if now is not None else time.time())
        today = now // DAY_SECONDS
        st = self.state
        if st.day != today and st.qty <= 0:
            self.state = st = DayState(day=today)

        action = decide(st, self._today_bars(now) if st.day == today else [], now)
        if action.kind == "buy":
            price = self.broker.get_price(self.symbol)
            qty = self.budget / price
            result = self.broker.place_order(Order(self.symbol, OrderSide.BUY, qty,
                                                   client_tag="orb_entry"))
            st.traded = True            # 下單失敗也不再重試,避免一天亂下多張
            if result.ok and result.filled_quantity <= 0:
                result = self._wait_fill(result)
            if result.ok and result.filled_quantity > 0:
                st.entry_price = result.avg_price or price
                st.qty = result.filled_quantity - self._base_fee(result.raw)
                st.entry_time = now
                st.entry_cost = st.entry_price * result.filled_quantity
                st.entry_order_id = result.order_id
                self.log(f"買進 {st.qty} {self.symbol} @ {st.entry_price}({action.reason})")
            else:
                self.log(f"買進失敗:{result.message}")
        elif action.kind == "sell":
            # OKX 現貨買單的手續費從買到的幣扣,可賣數量會比成交量少一點。
            sell_qty = min(st.qty, self._free_base())
            if sell_qty <= 0:
                self.log("帳戶裡已經沒有可賣的幣,視為已平倉")
                st.qty = 0.0
                st.save(self.state_path)
                return action
            result = self.broker.place_order(Order(self.symbol, OrderSide.SELL, sell_qty,
                                                   client_tag="orb_exit"))
            if result.ok:
                exit_price = result.avg_price or self.broker.get_price(self.symbol)
                proceeds = exit_price * sell_qty
                fee = self._quote_fee(result.raw, proceeds)
                # st.qty 已扣掉買單從幣收的手續費,成本用實際付出的 USDT 計算
                pnl = proceeds - fee - st.entry_cost
                _append_trade(self.trades_path, {
                    "symbol": self.symbol.replace("/", ""),
                    "side": "long",
                    "entry_time": _iso(st.entry_time or now),
                    "exit_time": _iso(now),
                    "entry_price": st.entry_price,
                    "exit_price": exit_price,
                    "quantity": sell_qty,
                    "pnl": round(pnl, 8),
                    "pnl_currency": self.symbol.split("/")[1],
                    "tag": f"ORB:{action.reason}",
                })
                self.log(f"賣出 {sell_qty} {self.symbol} @ {exit_price}({action.reason})")
                st.qty = 0.0
                if st.day != today:
                    self.state = st = DayState(day=today)
            else:
                self.log(f"賣出失敗,下一輪重試:{result.message}")
        st.save(self.state_path)
        return action

    def run_forever(self, poll_seconds: int = 30) -> None:
        self.log(f"開始執行(OKX 模擬盤){self.symbol},每筆 {self.budget} USDT。按 Ctrl+C 結束。")
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 — 網路錯誤等,下一輪再試
                self.log(f"本輪發生錯誤,稍後重試:{type(exc).__name__}: {exc}")
            time.sleep(poll_seconds)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="OKX 模擬盤當沖:開盤 1 小時區間突破")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--budget", type=float, default=100.0, help="每筆進場金額(USDT)")
    parser.add_argument("--state", default="okx_orb_state.json")
    parser.add_argument("--trades", default="okx_orb_trades.csv")
    parser.add_argument("--poll", type=int, default=30, help="每幾秒檢查一次")
    args = parser.parse_args(argv)

    missing = [k for k in ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE")
               if not os.environ.get(k)]
    if missing:
        print("⛔ 缺少環境變數:" + ", ".join(missing) + "(請用 OKX 模擬交易的 API key)")
        return 2

    try:
        import ccxt  # noqa: F401
    except ImportError:
        print('⛔ 尚未安裝 ccxt。請先執行:pip install -e ".[okx]"')
        return 2

    OKXBroker = load_okx_broker_class()
    broker = OKXBroker(os.environ["OKX_API_KEY"], os.environ["OKX_API_SECRET"],
                       os.environ["OKX_API_PASSPHRASE"], demo=True)
    broker.connect()
    broker.confirm_live_trading(i_understand_the_risk=True)   # 模擬盤:假錢
    symbol = broker.normalize_symbol(args.symbol)
    runner = OrbRunner(broker, symbol, args.budget, Path(args.state), Path(args.trades))
    try:
        runner.run_forever(args.poll)
    except KeyboardInterrupt:
        print("已停止。持倉狀態已存檔,重開會接續。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
