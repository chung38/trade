# -*- coding: utf-8 -*-
"""第 9 輪(SEO/AEO/E-E-A-T + 統計語意)回歸測試。

守住的東西:
  1. 誠實紀律 lint:絕對化/定罪式文案(「100% 是詐騙」「必賠」「注定」…)
     一經清除就不得再回來 —— 這些字眼在 YMYL 題材是信任毒藥,
     也違反本專案「機率語氣、不定罪」的鐵律。
  2. bootstrap p 值的置中語意與型一誤差校準。
  3. E-E-A-T 交付物存在且含關鍵錨點(官方連結、免責、可重現實驗)。
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

# 對外文案禁用清單:絕對化、定罪式、未校準百分比。
# 注意:這掃的是「我們自己的斷言」,所以排除 tests/(測試輸入會引用騙徒話術)。
BANNED = [
    "100% 是詐騙", "100%是詐騙",
    "數學上注定", "注定虧損",
    "必賠",
    "玩越久、賠越多",
    "沒有例外。",
    "鐵證",
    "不可能的夏普",
    # 第 9 輪複核追加(codex R9-05):機率語境裡的決定論詞
    "必然出現", "一定會有人", "一定會出現",
    "從來不存在",
    "幾乎沒有例外",
    "八成是雜訊",
    "不可能長期持續",
]

SCAN_TARGETS = (
    list(ROOT.glob("*.md"))
    + list(ROOT.glob("*.txt"))          # llms.txt / requirements.txt
    + list(ROOT.glob("*.cff"))
    + list((ROOT / "docs").glob("*.md"))
    + list((ROOT / ".claude").rglob("*.md"))
    + list((ROOT / "experiments").glob("*.py"))
    + [p for p in (ROOT / "core").rglob("*.py")]
)


def test_no_absolutist_wording_anywhere():
    """對外文案不得出現絕對化/定罪式字眼(第 9 輪 E-E-A-T 清理)。"""
    hits = []
    for f in SCAN_TARGETS:
        text = io.open(f, encoding="utf-8").read()
        for phrase in BANNED:
            if phrase in text:
                hits.append(f"{f.relative_to(ROOT)}: 「{phrase}」")
    assert not hits, "絕對化文案回歸:\n  " + "\n  ".join(hits)


def test_bootstrap_p_is_null_centered():
    """p 值必須來自 H0 置中重抽:全正樣本的 p 應極小,但不是靠「均值<=0 比例」。

    構造一個右偏樣本:置中重抽的 p 與舊法(未置中 tail)會不同 ——
    這裡驗證新語意的行為特徵:觀察均值越極端,p 越小;
    且對零期望資料,p 不會系統性偏小(由校準測試把關)。
    """
    from core.verdict.statistics import test_expectancy_positive

    strong = [100.0] * 25 + [-50.0] * 5
    weak = [10.0] * 16 + [-10.0] * 14
    r_strong = test_expectancy_positive(strong, n_bootstrap=2000, seed=7)
    r_weak = test_expectancy_positive(weak, n_bootstrap=2000, seed=7)
    assert r_strong.p_value_bootstrap < r_weak.p_value_bootstrap
    assert r_strong.p_value_bootstrap < 0.01


def test_bootstrap_type1_quick_calibration():
    """零期望資料的誤判率應接近 α(粗校準,完整版在 experiments/)。"""
    import random
    from core.verdict.statistics import test_expectancy_positive

    rng = random.Random(9)
    hits = 0
    trials = 120
    for i in range(trials):
        data = [rng.gauss(0, 100) for _ in range(35)]
        r = test_expectancy_positive(data, n_bootstrap=300, seed=i)
        if r.p_value_bootstrap < 0.05 and r.mean > 0:
            hits += 1
    rate = hits / trials
    assert 0.0 <= rate <= 0.13, f"型一誤差 {rate:.1%} 偏離 α=5% 過多"


def test_eeat_deliverables_exist_with_anchors():
    """E-E-A-T/AEO 交付物存在且含關鍵錨點。"""
    faq = io.open(ROOT / "docs" / "faq.md", encoding="utf-8").read()
    assert "165.npa.gov.tw" in faq and "不是司法鑑定" in faq
    assert "87%" in faq  # 「為什麼不說 87%」的自我解釋題必須在

    llms = io.open(ROOT / "llms.txt", encoding="utf-8").read()
    assert llms.startswith("# ") and "## Docs" in llms

    cff = io.open(ROOT / "CITATION.cff", encoding="utf-8").read()
    assert "cff-version" in cff and "社群化名" in cff

    readme = io.open(ROOT / "README.md", encoding="utf-8").read()
    assert "docs/faq.md" in readme and "165" in readme
    assert "English TL;DR" in readme
    assert "不是**立案法人" in readme or "不是**立案法人或" in readme.replace("\n", "")

    for name in ("CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md"):
        assert (ROOT / name).exists(), name

    for script in ("exp_multiple_comparison.py", "exp_survivorship.py",
                   "exp_bootstrap_calibration.py"):
        import py_compile
        py_compile.compile(str(ROOT / "experiments" / script), doraise=True)


def test_oos_report_discloses_hindsight_premise():
    """樣本外驗證必須揭露「規則若看著全段歷史調整,結論偏樂觀」前提。"""
    from datetime import datetime, timedelta
    from core.backtest.validate import holdout_validate
    from core.models import Market, Side, Trade, TradeLog

    trades = [Trade(symbol="X", market=Market.US_STOCK, side=Side.LONG,
                    entry_time=datetime(2026, 1, 1) + timedelta(days=i),
                    exit_time=datetime(2026, 1, 1) + timedelta(days=i),
                    entry_price=100, exit_price=101, quantity=10, fees=0,
                    pnl=(50 if i % 3 else -30)) for i in range(40)]
    rep = holdout_validate(TradeLog(trades=trades, source="t", account_label="t"),
                           n_bootstrap=300)
    assert any("前提提醒" in x for x in rep.interpretation)


def test_gambling_verdict_discloses_uncertainty():
    """負期望但 CI 上界 > 0 時,裁決要聲明「保守原則」而非「已證明必輸」。"""
    import random
    from core.models import Market, Side, Trade, TradeLog
    from core.verdict.judge import judge
    from datetime import datetime, timedelta

    rng = random.Random(3)
    trades = [Trade(symbol="X", market=Market.US_STOCK, side=Side.LONG,
                    entry_time=datetime(2026, 1, 1) + timedelta(days=i),
                    exit_time=datetime(2026, 1, 1) + timedelta(days=i),
                    entry_price=100, exit_price=100, quantity=10, fees=0,
                    pnl=rng.gauss(-5, 200)) for i in range(35)]
    log = TradeLog(trades=trades, source="t", account_label="t")
    v = judge(log, n_bootstrap=500)
    if v.metrics.expectancy < 0 and v.significance.ci_high > 0:
        assert any("保守原則" in r for r in v.reasons)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  OK {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)}")
    sys.exit(1 if failed else 0)
