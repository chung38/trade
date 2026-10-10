# -*- coding: utf-8 -*-
"""第 8 輪(裁決/計算/成本核心 + 新手實測 + 券商層)確認問題的回歸測試。"""
from __future__ import annotations

import math
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.metrics.performance import compute_metrics, fmt_ratio  # noqa: E402
from core.models import Market, Side, Trade, TradeLog  # noqa: E402


def _t(pnl, entry_price=1.0, qty=100.0, offset=0):
    d = datetime(2026, 1, 1) + timedelta(minutes=offset)
    exit_price = entry_price + pnl / qty
    return Trade(symbol="X", market=Market.US_STOCK, side=Side.LONG,
                 entry_time=d, exit_time=d, entry_price=entry_price,
                 exit_price=exit_price, quantity=qty, fees=0.0, pnl=pnl)


def test_drawdown_pct_no_lookahead():
    """回撤 % 必須用當下高水位:先虧 90 再暴賺,回撤是 90%,不是 8.9%。

    codex R8-PERF-001 的實例:資本代理 100,第一筆 -90(當時 peak=0,
    dd=90/(100+0)=90%),第二筆 +1000。舊算法用最終峰值 910 稀釋成 8.9%。
    """
    trades = [_t(-90.0, entry_price=1.0, qty=100.0, offset=0),
              _t(1000.0, entry_price=1.0, qty=100.0, offset=1)]
    m = compute_metrics(TradeLog(trades=trades, source="t", account_label="t"))
    assert m.max_drawdown == 90.0
    assert abs(m.max_drawdown_pct - 0.9) < 1e-9, \
        f"前視偏差:得到 {m.max_drawdown_pct:.1%},應為 90%"


def test_all_win_ratios_are_inf_not_zero():
    """全勝樣本的盈虧比/獲利因子/夏普是「不適用(∞)」,不是 0(最差)。"""
    trades = [_t(100.0, offset=i) for i in range(30)]
    m = compute_metrics(TradeLog(trades=trades, source="t", account_label="t"))
    assert math.isinf(m.profit_factor) and m.profit_factor > 0
    assert math.isinf(m.payoff_ratio) and m.payoff_ratio > 0
    assert m.expectancy == 100.0
    # largest_loss 不得填最小獲利
    assert m.largest_loss == 0.0 and m.largest_win == 100.0
    # 無虧損時 R 未定義,不編數字
    assert m.r_multiples == []
    # JSON 不得出現 Infinity
    d = m.as_dict()
    assert d["profit_factor"] is None and d["payoff_ratio"] is None
    # 顯示層不得印 0 或 inf 字樣
    assert "∞" in fmt_ratio(m.profit_factor)


def test_severe_drawdown_blocks_green_verdict():
    """腰斬級回撤的策略不得拿「具統計優勢」綠色裁決(P0 降級洞)。"""
    from core.verdict.judge import judge, VerdictLevel

    # 40 筆:先建立峰值,再一筆 -60% 回撤,之後穩定小賺 → 總體顯著為正
    trades = [_t(50.0, entry_price=1.0, qty=100.0, offset=i) for i in range(10)]
    trades.append(_t(-400.0, entry_price=1.0, qty=100.0, offset=10))
    trades += [_t(60.0, entry_price=1.0, qty=100.0, offset=11 + i) for i in range(30)]
    log = TradeLog(trades=trades, source="t", account_label="t")
    m = compute_metrics(log)
    assert m.max_drawdown_pct > 0.5, f"前置條件:回撤要腰斬(得到 {m.max_drawdown_pct:.0%})"
    v = judge(log, metrics=m, n_bootstrap=2000)
    if v.significance.is_significant and v.metrics.expectancy > 0:
        assert v.level != VerdictLevel.STATISTICAL_EDGE, \
            "腰斬回撤 + 顯著為正:必須降級,不得給綠色"
        assert v.should_discourage


def test_zero_expectancy_wording_honest():
    """期望值恰為 0:不得說「帳面上賺錢」。"""
    from core.verdict.judge import judge

    trades = ([_t(100.0, offset=i) for i in range(15)]
              + [_t(-100.0, offset=15 + i) for i in range(15)])
    log = TradeLog(trades=trades, source="t", account_label="t")
    v = judge(log, n_bootstrap=1000)
    assert v.metrics.expectancy == 0.0
    assert "帳面上賺錢" not in v.headline, v.headline


def test_pnl_only_drawdown_pct_flagged_unreliable():
    """pnl-only 資料沒有資本基準:回撤 % 標記不可靠,報告寫「無法計算」。"""
    trades = [Trade(symbol="X", market=Market.US_STOCK, side=Side.LONG,
                    entry_time=datetime(2026, 1, 1) + timedelta(minutes=i),
                    exit_time=datetime(2026, 1, 1) + timedelta(minutes=i),
                    entry_price=0.0, exit_price=0.0, quantity=0.0, fees=0.0,
                    pnl=p) for i, p in enumerate((100.0, -50.0, 30.0))]
    m = compute_metrics(TradeLog(trades=trades, source="t", account_label="t"))
    assert not m.drawdown_pct_reliable
    assert m.max_drawdown == 50.0


def test_us_cost_includes_taf_and_updated_sec():
    """美股成本:SEC 0.0000206(2026-04 起)+ FINRA TAF(賣出每股,有上限)。"""
    from core.ingest.costs import estimate_round_trip_cost

    c = estimate_round_trip_cost(Market.US_STOCK, Side.LONG, 100, 110, 100)
    # 手算:滑價 21000×0.0005=10.5 + SEC 11000×0.0000206=0.2266 + TAF 100×0.000195=0.0195
    assert abs(c - 10.7461) < 0.001, c
    # TAF 上限:賣 10 萬股 → 每股 0.000195×100000=19.5 → 封頂 9.79
    c2 = estimate_round_trip_cost(Market.US_STOCK, Side.LONG, 1, 1, 100_000)
    taf_part = c2 - (200_000 * 0.0005) - (100_000 * 0.0000206)
    assert abs(taf_part - 9.79) < 0.01, taf_part


def test_loader_reads_big5_csv():
    """繁中 Excel 存的 ANSI(Big5)CSV 必須能讀(新手照文件操作的常見路徑)。"""
    from core.ingest.loader import load_trades

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "big5.csv"
        content = ("代號,方向,進場時間,出場時間,進場價,出場價,數量,交易成本,策略\n"
                   "2330,買,2025-01-03,2025-02-10,1000,1080,1000,0,季線突破\n")
        p.write_bytes(content.encode("cp950"))
        log = load_trades(str(p))
        assert len(log.trades) == 1
        assert log.trades[0].tag == "季線突破"


def test_config_credentials_match_broker_signature():
    """config.example.yaml 的 credentials 欄位必須對應該券商建構子簽名。"""
    from core.scaffold.templates import config_yaml
    from core.scaffold.generator import ScaffoldOptions

    y = config_yaml(ScaffoldOptions(project_name="t", broker="ibkr",
                                    symbols=["AAPL"]), False, "unknown")
    assert "host:" in y and "client_id:" in y and "api_secret" not in y
    y2 = config_yaml(ScaffoldOptions(project_name="t", broker="okx",
                                     symbols=["BTCUSDT"]), False, "unknown")
    assert "passphrase:" in y2


def test_required_sample_size_handles_inf_payoff():
    """有打平但無虧損的樣本(payoff=inf)不得讓所需樣本數變 NaN。"""
    from core.verdict.statistics import required_sample_size

    n = required_sample_size(0.8, float("inf"))
    assert isinstance(n, int) and n > 0


# ── 第 8 輪複核(兩模型驗收)確認問題的回歸 ─────────────────────────


def test_drawdown_capital_base_is_causal():
    """未來才放大的部位不得稀釋早期回撤(複核抓到的第二層前視偏差)。

    第一筆部位 100 元虧 90(因果口徑 90%),之後 39 筆用 100 萬部位 ——
    舊算法拿整段最大部位當分母,會把 90% 稀釋成 0.009%。
    """
    ts = [_t(-90, 1, 100, 0)] + [_t(100, 10000, 100, i + 1) for i in range(39)]
    m = compute_metrics(TradeLog(trades=ts, source="t", account_label="t"))
    assert abs(m.max_drawdown_pct - 0.9) < 1e-9, \
        f"未來部位稀釋了早期回撤:{m.max_drawdown_pct:.4%}"


def test_json_no_infinity_anywhere():
    """全勝紀錄的完整 as_dict(含 OOS/per-tag)不得含 Infinity/NaN。"""
    import json as _json
    from core.analyzer import analyze_log

    trades = [_t(100.0, offset=i) for i in range(30)]
    for i, t in enumerate(trades):
        t.tag = "全勝策略"
    r = analyze_log(TradeLog(trades=trades, source="t", account_label="t"),
                    n_bootstrap=200)
    _json.dumps(r.as_dict(), allow_nan=False)  # 有漏就 ValueError


def test_severe_drawdown_needs_reliable_pct():
    """pnl-only 資料的回撤 %「無法計算」:不得同時拿它發 severe_drawdown。"""
    from core.verdict.judge import _scan_red_flags
    from core.verdict.statistics import test_expectancy_positive

    trades = [Trade(symbol="X", market=Market.US_STOCK, side=Side.LONG,
                    entry_time=datetime(2026, 1, 1), exit_time=datetime(2026, 1, 1),
                    entry_price=0.0, exit_price=0.0, quantity=0.0, fees=0.0,
                    pnl=p) for p in ([100.0] * 5 + [-60.0] + [10.0] * 24)]
    m = compute_metrics(TradeLog(trades=trades, source="t", account_label="t"))
    assert not m.drawdown_pct_reliable
    sig = test_expectancy_positive([t.pnl for t in trades], n_bootstrap=200)
    flags = _scan_red_flags(m, sig)
    assert not any(f.code == "severe_drawdown" for f in flags), \
        "報告說回撤 % 無法計算,裁決卻拿它定罪 —— 通道不一致"


def test_taf_low_price_exception():
    """成交價低於每股 TAF 費率時不收 TAF(FINRA 低價例外)。"""
    from core.ingest.costs import CostModel

    m = CostModel(commission_rate=0, commission_min=0, tax_rate=0,
                  slippage_rate=0, sell_per_share_fee=0.000195,
                  sell_per_share_fee_cap=9.79)
    assert m.estimate(0.0001, 100, is_sell=True) == 0.0
    assert m.estimate(1.0, 100, is_sell=True) > 0.0


def test_day_trade_model_keeps_new_fields():
    """當沖減半複製 CostModel 時不得清零新欄位(dataclasses.replace)。"""
    import dataclasses
    from core.ingest.costs import CostModel

    m = CostModel(commission_rate=0.001425, commission_min=20, tax_rate=0.003,
                  slippage_rate=0.0005, sell_per_share_fee=0.1,
                  sell_per_share_fee_cap=5.0)
    m2 = dataclasses.replace(m, tax_rate=0.0015)
    assert m2.sell_per_share_fee == 0.1 and m2.sell_per_share_fee_cap == 5.0


def test_cp950_read_is_disclosed_in_source():
    """cp950 回退讀取必須在 source 揭露(亂碼時使用者才有線索)。"""
    from core.ingest.loader import load_trades

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "b5.csv"
        p.write_bytes(
            ("symbol,net_pnl,pnl_currency,tag\n2330,100,TWD,測試策略\n").encode("cp950"))
        log = load_trades(str(p))
        assert "cp950" in log.source or "Big5" in log.source
        # 合法 UTF-8 檔不得標 cp950
        p2 = Path(td) / "u8.csv"
        p2.write_text(
            "symbol,net_pnl,pnl_currency,tag\n2330,100,TWD,測試策略\n",
            encoding="utf-8",
        )
        log2 = load_trades(str(p2))
        assert "cp950" not in log2.source
        assert log2.trades[0].tag == "測試策略"


def test_credential_fields_cover_all_brokers():
    """_CREDENTIAL_FIELDS 必須涵蓋全部券商(新增券商漏建映射要被抓到)。"""
    from core.broker import BROKER_TEMPLATES
    from core.scaffold.templates import _CREDENTIAL_FIELDS

    assert set(_CREDENTIAL_FIELDS.keys()) == set(BROKER_TEMPLATES.keys()), \
        f"映射與註冊表不同步:{set(_CREDENTIAL_FIELDS) ^ set(BROKER_TEMPLATES)}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ❌ {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 通過")
    sys.exit(1 if failed else 0)
