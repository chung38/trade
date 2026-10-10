# -*- coding: utf-8 -*-
"""第 7 輪(scaffold 實測 + 文件查核 + 冷區辯論)確認問題的回歸測試。"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.markets import infer_market  # noqa: E402
from core.models import Market, Side, Trade, TradeLog  # noqa: E402


def test_bare_coin_symbols_do_not_hijack_us_tickers():
    """SOL(NYSE)、ETHA(ETF)是美股;裸幣名模稜兩可回 UNKNOWN。"""
    assert infer_market("SOL") == Market.UNKNOWN, "裸 SOL 與美股代號衝突,應 UNKNOWN"
    assert infer_market("BTC") == Market.UNKNOWN
    assert infer_market("ETHA") == Market.US_STOCK, "ETHA 是真實美股 ETF"
    assert infer_market("SOLAR") == Market.US_STOCK
    # 有計價幣/分隔符的照常判 crypto
    for s in ("BTCUSDT", "ETHUSD", "BTC-USD", "BTC/USDT", "SOLUSDT"):
        assert infer_market(s) == Market.CRYPTO, s
    # hint 仍可強制
    assert infer_market("BTC", Market.CRYPTO) == Market.CRYPTO


def test_trade_rejects_exit_before_entry():
    """出場早於進場是髒資料,必須 ValueError,不得變成負持倉天數。"""
    try:
        Trade(symbol="2330", market=Market.TW_STOCK, side=Side.LONG,
              entry_time=datetime(2025, 3, 1), exit_time=datetime(2025, 1, 1),
              entry_price=100, exit_price=110, quantity=1000)
        raise AssertionError("應 raise ValueError")
    except ValueError as e:
        assert "早於" in str(e)


def test_loader_skips_bad_time_order_row():
    """loader 遇到出場早於進場的列:略過 + skip_reasons 揭露,不汙染統計。"""
    from core.ingest.loader import load_trades

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        p.write_text(
            "symbol,side,entry_time,exit_time,entry_price,exit_price,quantity,fees\n"
            "2330,long,2025-03-01,2025-01-01,100,110,1000,0\n"
            "2330,long,2025-01-01,2025-03-01,100,110,1000,0\n",
            encoding="utf-8")
        log = load_trades(str(p))
        assert len(log.trades) == 1
        assert "早於" in log.source or "略過原因" in log.source


def test_loader_refuses_unknown_multiplier_derivative_without_pnl():
    """台期/選擇權乘數未知且沒給 pnl:用 1.0 推算是錯的數字,必須略過。"""
    from core.ingest.loader import load_trades

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f.csv"
        # ZZZF202606 不在乘數白名單;hint 強制期貨市場
        p.write_text(
            "symbol,side,entry_time,exit_time,entry_price,exit_price,quantity\n"
            "ZZZF202606,long,2025-01-01,2025-01-02,100,110,1\n"
            "ZZZF202606,long,2025-01-01,2025-01-02,100,110,1\n",
            encoding="utf-8")
        try:
            load_trades(str(p), market_hint=Market.TW_FUTURES)
            raise AssertionError("全部列都應被略過 → ValueError")
        except ValueError as e:
            assert "乘數未知" in str(e)
        # 直接給 pnl 就照收(使用者自己算好的數字)
        p2 = Path(td) / "f2.csv"
        p2.write_text(
            "symbol,entry_time,exit_time,net_pnl,pnl_currency\n"
            "ZZZF202606,2025-01-01,2025-01-02,2000,TWD\n",
            encoding="utf-8")
        log = load_trades(str(p2), market_hint=Market.TW_FUTURES)
        assert len(log.trades) == 1 and log.trades[0].pnl == 2000.0


def _mk_log(tags, pnls):
    trades = []
    for i, (tag, pnl) in enumerate(zip(tags, pnls)):
        trades.append(Trade(
            symbol="2330", market=Market.TW_STOCK, side=Side.LONG,
            entry_time=datetime(2025, 1, 1 + i), exit_time=datetime(2025, 1, 2 + i),
            entry_price=100, exit_price=100, quantity=1000, fees=0.0,
            pnl=pnl, tag=tag))
    return TradeLog(trades=trades, source="test", account_label="t")


def test_scam_warning_not_triggered_by_legit_group_tags():
    """「族群輪動」是正常策略標籤,不得被裸「群」子字串誤判成跟單詐騙。"""
    from core.antiscam.signals import scam_warnings_for
    from core.metrics.performance import compute_metrics
    from core.strategy.profiler import profile_strategy

    log = _mk_log(["族群輪動"] * 12, [-100] * 12)
    warns = scam_warnings_for(compute_metrics(log), profile_strategy(log))
    assert not any("假飆股群" in w for w in warns), f"族群輪動被誤判:{warns}"

    # 真正的跟單標籤仍要抓到,且措辭不得宣稱「你的虧損交易中」(未驗證的歸因)
    log2 = _mk_log(["VIP群老師帶單"] * 12, [-100] * 12)
    warns2 = scam_warnings_for(compute_metrics(log2), profile_strategy(log2))
    assert any("假飆股群" in w for w in warns2)
    assert not any("你的虧損交易中" in w for w in warns2)


def test_high_winrate_trap_needs_min_sample():
    """<10 筆時勝率沒有意義,不得輸出「高勝率話術陷阱」定錨敘事。"""
    from core.antiscam.signals import scam_warnings_for
    from core.metrics.performance import compute_metrics
    from core.strategy.profiler import profile_strategy

    log = _mk_log([None] * 5, [10, 10, 10, 10, -100])  # 80% 勝率,5 筆,EV<0
    warns = scam_warnings_for(compute_metrics(log), profile_strategy(log))
    assert not any("高勝率" in w for w in warns), f"5 筆樣本不該有話術警語:{warns}"


def test_scam_check_rejects_garbage_input_and_eof():
    """看不懂的輸入必須重問(不可當「否」);EOF 中斷不給結論。"""
    from core.antiscam.checklist import CHECK_ITEMS, run_scam_check

    # 第一題先餵垃圾再答 y,其餘答 n → 垃圾輸入必須觸發重問
    feeds = ["也許吧", "y"] + ["n"] * (len(CHECK_ITEMS) - 1)
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return feeds.pop(0)

    result = run_scam_check(input_fn=fake_input, output_fn=lambda *_: None)
    assert result is not None
    assert len(prompts) == len(CHECK_ITEMS) + 1, "垃圾輸入應觸發同題重問"
    assert result.score > 0, "第一題最終答 y,分數應計入"

    # EOF → None(不給半套結論)
    def eof_input(prompt):
        raise EOFError

    out = []
    r2 = run_scam_check(input_fn=eof_input, output_fn=out.append)
    assert r2 is None
    assert any("不給任何風險結論" in str(x) for x in out)


def test_profiler_median_even_count():
    """偶數筆持倉天數的中位數要取中央兩值平均(1 天與 9 天 → 5 天)。"""
    from core.strategy.profiler import profile_strategy

    trades = [
        Trade(symbol="2330", market=Market.TW_STOCK, side=Side.LONG,
              entry_time=datetime(2025, 1, 1), exit_time=datetime(2025, 1, 2),
              entry_price=100, exit_price=101, quantity=1000, fees=0.0, pnl=100),
        Trade(symbol="2330", market=Market.TW_STOCK, side=Side.LONG,
              entry_time=datetime(2025, 2, 1), exit_time=datetime(2025, 2, 10),
              entry_price=100, exit_price=101, quantity=1000, fees=0.0, pnl=100),
    ]
    prof = profile_strategy(TradeLog(trades=trades, source="t", account_label="t"))
    assert prof.median_holding_days == 5.0, f"得到 {prof.median_holding_days}"


def test_report_discloses_uncovered_costs():
    """crypto 紀錄的報告必須出現「未涵蓋成本」誠實聲明(資金費率)。"""
    from core.analyzer import analyze_file
    from core.cli import _example_path

    path, market = _example_path("crypto")
    result = analyze_file(path, market_hint=market, n_bootstrap=500)
    assert "未涵蓋的成本" in result.text_report
    assert "資金費率" in result.text_report


# ── 第 7 輪複核(兩模型驗收)確認問題的回歸 ─────────────────────────


def test_coin_quoted_pairs_still_crypto():
    """幣幣對(ETHBTC/SOLBTC)不得因裸幣修復而變成 crypto 假陰性。"""
    for s in ("ETHBTC", "SOLBTC", "BNBBTC", "ETH-BTC", "SOL_USDT"):
        assert infer_market(s) == Market.CRYPTO, s
    # 修復不得重新引入前綴劫持
    assert infer_market("ETHA") == Market.US_STOCK
    assert infer_market("SOLAR") == Market.US_STOCK


def test_industry_group_tag_not_scam():
    """「產業群組輪動」是正常策略標籤,「群組」泛詞不得觸發詐騙警語。"""
    from core.antiscam.signals import scam_warnings_for
    from core.metrics.performance import compute_metrics
    from core.strategy.profiler import profile_strategy

    log = _mk_log(["產業群組輪動"] * 12, [-100] * 12)
    warns = scam_warnings_for(compute_metrics(log), profile_strategy(log))
    assert not any("假飆股群" in w for w in warns), f"產業群組被誤判:{warns}"


def test_scaffold_symbols_sentinel_removed():
    """--symbols 預設必須是 None:明確輸入 AAPL 的人不得被當成沒指定。"""
    from core.cli import _build_parser

    parser = _build_parser()
    a1 = parser.parse_args(["scaffold", "--name", "x"])
    assert a1.symbols is None, "未指定時應為 None(不是哨兵字串 AAPL)"
    a2 = parser.parse_args(["scaffold", "--name", "x", "--symbols", "AAPL"])
    assert a2.symbols == "AAPL"


def test_scaffold_main_template_time_outer_loop():
    """模板主迴圈必須時間外圈:逐檔跑完整段歷史會讓權益曲線時間穿越。"""
    from core.charts.registry import get_chart_lib
    from core.scaffold.generator import ScaffoldOptions
    from core.scaffold.templates import main_py

    src = main_py(ScaffoldOptions(project_name="t", symbols=["AAA", "BBB"]),
                  get_chart_lib("lightweight"), None, False)
    assert "for i in range(n_bars):" in src, "應為時間外圈"
    assert "histories = {" in src
    assert "getattr(broker, 'name'" in src, "摘要應用實際 broker 名稱"
    compile(src, "<scaffold-main>", "exec")


def test_scaffold_templates_no_stale_broker_hints():
    """config 券商註解動態化;broker_setup 不得硬給 (api_key, api_secret) 呼叫。"""
    from core.broker import BROKER_TEMPLATES
    from core.scaffold.templates import _broker_keys_comment, broker_setup_py, config_yaml
    from core.scaffold.generator import ScaffoldOptions

    comment = _broker_keys_comment()
    for key in BROKER_TEMPLATES:
        assert key in comment, f"動態清單缺 {key}"
    opts = ScaffoldOptions(project_name="t", symbols=["AAPL"], broker="ibkr")
    setup_src = broker_setup_py(opts, BROKER_TEMPLATES["ibkr"])
    assert 'creds["api_key"], creds["api_secret"])' not in setup_src, \
        "IBKR 建構子是 host/port/client_id,不得硬給金鑰簽名範例"


def test_chart_docstrings_honest_about_cdn():
    """CDN 圖表庫不得自稱「自包含」(離線不渲染)。"""
    from core.charts.registry import get_chart_lib

    for key in ("lightweight", "echarts"):
        assert "自包含" not in get_chart_lib(key).module_code, key


def test_html_report_discloses_uncovered_costs():
    """HTML 報告也要有未涵蓋成本警語 —— 分享出去的通道不能比終端不誠實。"""
    from core.analyzer import analyze_file
    from core.cli import _example_path
    from core.report_html import render_html_report

    path, market = _example_path("crypto")
    result = analyze_file(path, market_hint=market, n_bootstrap=500)
    html = render_html_report(result)
    assert "未涵蓋的成本" in html
    assert "資金費率" in html


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
