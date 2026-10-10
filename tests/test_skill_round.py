# -*- coding: utf-8 -*-
"""技能層優化輪的回歸測試。

守住的教訓:
  1. 骨架編譯:_entry_logic_hint 曾硬編 8 格縮排,導致 9 種組合有 6 種
     產出 IndentationError 的骨架 —— 而且從來沒有測試真的去編譯過產出。
     這裡把 3 個內建範例 × 3 個框架全部產一次、全部 compile() 一次。
  2. --full 一鍵健檢:主報告後必須接月報趨勢與風險情境(或誠實的略過訊息),
     HTML 版必須帶「情境非預測」警語,不能只給數字。
  3. 下一步導流:沒帶 --full 時要主動告訴使用者 trend / risk-sim 的存在。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyzer import analyze_file  # noqa: E402
from core.cli import _example_path  # noqa: E402
from core.strategy.skeleton import generate_skeleton  # noqa: E402

_FRAMEWORKS = ("backtrader", "vectorbt", "generic")
_EXAMPLE_KEYS = ("tw", "us", "crypto")


def _run_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "core.cli", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path(__file__).resolve().parent.parent),
        # 覆寫值放在展開之後,否則外部環境同名變數會蓋掉測試指定值
        env={**__import__("os").environ,
             "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        timeout=600,
    )


def test_all_skeleton_combos_compile():
    """3 範例 × 3 框架 = 9 種骨架,每一種都必須是合法 Python。"""
    for key in _EXAMPLE_KEYS:
        path, market = _example_path(key)
        result = analyze_file(path, market_hint=market, n_bootstrap=500)
        for fw in _FRAMEWORKS:
            code = generate_skeleton(result.profile, result.verdict, framework=fw)
            try:
                compile(code, f"<skeleton:{key}:{fw}>", "exec")
            except SyntaxError as e:  # pragma: no cover - 失敗時給可讀訊息
                raise AssertionError(
                    f"骨架無法編譯:example={key} framework={fw} — {e}"
                ) from e


def test_skeleton_entry_hint_indented_correctly():
    """進場提示在 backtrader 是 12 格、generic 是 4 格(曾經全錯成 8 格)。"""
    path, market = _example_path("tw")
    result = analyze_file(path, market_hint=market, n_bootstrap=500)
    bt_code = generate_skeleton(result.profile, result.verdict, framework="backtrader")
    ge_code = generate_skeleton(result.profile, result.verdict, framework="generic")
    assert "\n            signal = False" in bt_code, "backtrader 提示應為 12 格縮排"
    assert "\n    signal = False" in ge_code, "generic 提示應為 4 格縮排"


def test_analyze_full_appends_trend_and_risk():
    """--full 必須在主報告後接趨勢與風險情境;範例 tw 為賭博 → exit 2。"""
    with tempfile.TemporaryDirectory() as td:
        html = str(Path(td) / "full.html")
        p = _run_cli("analyze", "--example", "tw", "--full",
                     "--equity", "500000", "--html", html)
        assert p.returncode == 2, f"tw 範例應勸退(exit 2),得到 {p.returncode}"
        assert "時間趨勢" in p.stdout, "--full 缺少月報趨勢區塊"
        assert "風險情境模擬" in p.stdout, "--full 缺少風險情境區塊"
        h = Path(html).read_text(encoding="utf-8")
        assert "月報趨勢" in h, "--full --html 應包含趨勢區塊"
        assert "風險情境模擬" in h, "--full --html 應包含風險情境區塊"
        assert "不是「預測」" in h, "HTML 風險區塊必須帶「情境非預測」警語"


def test_analyze_without_full_suggests_trend_and_risksim():
    """沒帶 --full 時,下一步導流必須提示 trend 與 risk-sim 的存在。"""
    p = _run_cli("analyze", "--example", "tw")
    assert "trend" in p.stdout, "下一步應導流 trend"
    assert "risk-sim" in p.stdout, "下一步應導流 risk-sim"
    assert "--full" in p.stdout, "下一步應提示 --full 一次看完"


def test_html_without_full_has_no_extra_blocks():
    """沒帶 --full 時,HTML 保持精簡 —— 不得出現趨勢/風險區塊。"""
    with tempfile.TemporaryDirectory() as td:
        html = str(Path(td) / "plain.html")
        p = _run_cli("analyze", "--example", "tw", "--html", html)
        assert p.returncode == 2
        h = Path(html).read_text(encoding="utf-8")
        assert "月報趨勢" not in h
        assert "風險情境模擬" not in h


# ── 第 6 輪辯論(gpt-5.6-sol + grok-4.5)確認的 bug 回歸測試 ──────────────


def test_loader_rejects_rows_without_computable_pnl():
    """空白 pnl 且價格不全的列必須略過,不得變成假打平/假獲利交易。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pnl_blank.csv"
        p.write_text(
            "symbol,net_pnl,pnl_currency\nAAPL,100,USD\nAAPL,,USD\nAAPL,-50,USD\n", encoding="utf-8")
        from core.ingest.loader import load_trades
        log = load_trades(str(p))
        assert len(log.trades) == 2, "空白 pnl 列應被略過,不是補 0"
        assert [t.pnl for t in log.trades] == [100.0, -50.0]

        # 進場價無法解析 → 不得把出場價整段當獲利
        p2 = Path(td) / "bad_price.csv"
        p2.write_text(
            "symbol,side,entry_price,exit_price,quantity,fees\n"
            "AAPL,long,abc,100,10,0\nAAPL,long,50,60,10,0\n", encoding="utf-8")
        log2 = load_trades(str(p2))
        assert len(log2.trades) == 1, "進場價爛掉的列應略過"
        assert log2.trades[0].pnl == 100.0

        # nan/inf 字串視同無法解析
        p3 = Path(td) / "nan_inf.csv"
        p3.write_text(
            "symbol,net_pnl,pnl_currency\nAAPL,nan,USD\nAAPL,inf,USD\nAAPL,5,USD\n", encoding="utf-8")
        log3 = load_trades(str(p3))
        assert [t.pnl for t in log3.trades] == [5.0]


def test_ruin_scenario_rejects_nan_inf_equity():
    """NaN 本金會讓爆倉比例假裝 0%、inf 會全爆 —— 必須直接拒絕。"""
    from core.montecarlo import simulate_ruin_scenario

    pnls = [100.0, -100.0] * 6
    for bad in (float("nan"), float("inf"), 0.0, -5.0):
        try:
            simulate_ruin_scenario(pnls, start_equity=bad)
            raise AssertionError(f"start_equity={bad} 應 raise ValueError")
        except ValueError:
            pass
    try:
        simulate_ruin_scenario(pnls + [float("nan")], start_equity=10000.0)
        raise AssertionError("含 NaN 的損益序列應 raise ValueError")
    except ValueError:
        pass


def test_analyze_full_equity_zero_is_friendly_error():
    """--equity 0 必須是友善錯誤(exit 1 + 中文訊息),不是 traceback。"""
    p = _run_cli("analyze", "--example", "tw", "--full", "--equity", "0")
    assert p.returncode == 1
    assert "Traceback" not in p.stderr, f"不該有 traceback:{p.stderr[-300:]}"
    assert "--equity" in p.stderr
    assert "有限數" in p.stderr, "錯誤訊息應為中文且說明約束"


def test_equity_without_full_warns():
    """--equity 沒搭 --full 時要提醒未生效,不能靜默忽略。"""
    p = _run_cli("analyze", "--example", "tw", "--equity", "500000")
    assert "未帶 --full" in p.stderr
    # 只是警告,分析本身必須照常完成(tw 範例=勸退,exit 2 且有裁決輸出)
    assert p.returncode == 2
    assert "賭博" in p.stdout


def test_format_fraction_never_rounds_open_interval_to_endpoints():
    """0.9996 不可顯示 100%、0.0004 不可顯示 0% —— 端點只留給真端點。"""
    import re
    from core.montecarlo import format_fraction as ff

    assert ff(1.0) == "100%" and ff(0.0) == "0%"
    for frac in (0.9996, 0.9999, 0.99951):
        out = ff(frac)
        assert not re.match(r"^100(\.0+)?%$", out), f"{frac} 顯示 {out}"
    for frac in (0.0004, 0.0001, 0.00049):
        out = ff(frac)
        assert not re.match(r"^0(\.0+)?%$", out), f"{frac} 顯示 {out}"
    assert ff(0.37) == "37%"
    # 複核輪抓到的邊界 bug:0.5%/99.5% 曾被字串攔截誤標成 <0.1%/>99.9%
    assert ff(0.005) == "0.5%", f"0.005 顯示 {ff(0.005)}"
    assert ff(0.995) == "99.5%", f"0.995 顯示 {ff(0.995)}"


def test_drawdown_display_uses_endpoint_discipline():
    """最大回撤欄也要守端點紀律:0.9996 的回撤不可顯示成 100%。"""
    import dataclasses
    from core.montecarlo import RuinScenario, render_scenario

    s = RuinScenario(
        n_future_trades=200, n_paths=5000, start_equity=100000.0,
        ruin_threshold=50000.0, ruin_fraction=0.5, median_final_equity=100.0,
        p05_final_equity=0.0, p95_final_equity=200.0,
        median_max_drawdown=0.004, p95_max_drawdown=0.9996,
        median_trade_at_ruin=None, losing_streak_10_prob=0.5,
        warnings=[], start_equity_inferred=False,
    )
    line = next(x for x in render_scenario(s).splitlines() if "最大回撤" in x)
    assert "100%" not in line, f"p95 回撤 0.9996 不可顯示 100%:{line}"
    assert "0%," not in line.replace("<0.1%", ""), f"中位回撤 0.004 不可顯示 0%:{line}"


def test_loader_accepts_zeroed_option():
    """歸零的選擇權(出場價 0)是合法交易,守門不得誤殺。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "zeroed.csv"
        p.write_text(
            "symbol,side,entry_price,exit_price,quantity,fees\n"
            "TXO18000C25A,long,50,0,1,0\n", encoding="utf-8")
        from core.ingest.loader import load_trades
        log = load_trades(str(p))
        assert len(log.trades) == 1, "出場價 0 的歸零商品應可載入"
        assert log.trades[0].pnl < 0


def test_gambling_wording_is_probabilistic():
    """裁決措辭不可用「注定/必然」的決定論語氣(樣本≠母體)。"""
    p = _run_cli("analyze", "--example", "tw")
    assert "注定" not in p.stdout, "gambling 裁決不可說「注定」"
    assert "玩越久、賠越多,這是數學" not in p.stdout


def test_full_html_discloses_skipped_scenario():
    """--full 但樣本 <10 筆:HTML 必須寫明「已略過+原因」,不得靜默消失。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "small.csv"
        rows = "\n".join(
            f"AAPL,long,2024-0{i%2+1}-0{i+1},2024-0{i%2+1}-1{i},100,10{i},10"
            for i in range(5)
        )
        csv.write_text(
            "symbol,side,entry_time,exit_time,entry_price,exit_price,quantity\n"
            + rows + "\n", encoding="utf-8")
        html = Path(td) / "r.html"
        p = _run_cli("analyze", str(csv), "--full", "--html", str(html))
        assert "已略過風險情境模擬" in p.stdout
        content = html.read_text(encoding="utf-8")
        assert "已略過風險情境模擬" in content
        assert "尚未被評估" in content


def test_full_json_contains_extras():
    """--full --json:JSON 必須含 full_extras(趨勢+風險情境)。"""
    import json as _json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "o.json"
        _run_cli("analyze", "--example", "tw", "--full",
                 "--equity", "500000", "--json", str(out))
        data = _json.loads(out.read_text(encoding="utf-8"))
        assert "full_extras" in data
        assert data["full_extras"]["trend"] is not None
        assert data["full_extras"]["risk_scenario"] is not None


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
