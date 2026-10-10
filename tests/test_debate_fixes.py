"""多方辯論(Codex + Grok + Claude)稽核後的修正 — 防退化測試。

每個測試對應一項經蒙地卡羅/構造案例驗證為真的缺陷,防止未來改壞。
執行: python tests/test_debate_fixes.py
"""

from __future__ import annotations

import random
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyzer import analyze_log  # noqa: E402
from core.broker import Order, OrderSide, PaperBroker  # noqa: E402
from core.models import Market, Side, Trade, TradeLog  # noqa: E402
from core.strategy.per_tag import per_tag_verdicts  # noqa: E402
from core.verdict.judge import judge  # noqa: E402
from core.verdict.statistics import (  # noqa: E402
    Z_ALPHA_ONE_SIDED,
    Z_POWER_80,
    required_sample_size,
    required_sample_size_from_pnls,
)


def _t(pnl, i, tag=None):
    d = datetime(2024, 1, 1) + timedelta(days=i)
    return Trade("X", Market.US_STOCK, Side.LONG, d, d + timedelta(days=1),
                 100, 100, 10, fees=0, pnl=pnl, tag=tag)


# ── P0: 報告不得謊稱「通過樣本外驗證」 ─────────────────────
def test_report_never_claims_oos_pass_when_it_failed():
    """樣本內顯著、但樣本外未延續時,報告不可宣稱通過樣本外驗證。"""
    rng = random.Random(7)
    pnls = [rng.gauss(60, 15) for _ in range(60)] + [rng.gauss(2, 40) for _ in range(25)]
    r = analyze_log(TradeLog([_t(p, i) for i, p in enumerate(pnls)]), n_bootstrap=1000)
    assert r.verdict.level.value == "statistical_edge"
    assert not r.out_of_sample.edge_persisted
    # 關鍵斷言:不可出現「通過了統計檢定與樣本外驗證」
    assert "通過了統計檢定與樣本外驗證" not in r.text_report
    assert "樣本外驗證尚未確認" in r.text_report


# ── P1: judge 不得硬寫「信賴區間完全落在 0 以上」 ──────────
def test_judge_no_false_ci_claim_when_ci_low_negative():
    """is_significant 為真但 ci_low<0 時,不可宣稱 CI 完全落在 0 以上。"""
    found = False
    for seed in range(300):
        rng = random.Random(seed)
        pnls = [rng.gauss(9, 42) for _ in range(45)]
        v = judge(TradeLog([_t(p, i) for i, p in enumerate(pnls)]), n_bootstrap=2000)
        s = v.significance
        if s.is_significant and s.ci_low < 0:
            found = True
            assert not any("完全落在 0 以上" in r for r in v.reasons), \
                "CI 下界為負卻宣稱完全落在 0 以上(對使用者說假話)"
            assert any("邊際等級" in r or "下界仍略低於 0" in r for r in v.reasons)
            break
    assert found, "未構造出 ci_low<0 且顯著的案例(請調整 seed 範圍)"


# ── P1: per-tag 不得對純隨機雜訊頒發優勢認證 ───────────────
def test_per_tag_never_certifies_random_noise():
    """K 個純隨機零期望策略,不得有任何一個被認證為具優勢。"""
    for K in (5, 10):
        rng = random.Random(K)
        trades = []
        for k in range(K):
            for i in range(12):
                trades.append(_t(rng.gauss(0, 50), k * 12 + i, tag=f"策略{k}"))
        tv = per_tag_verdicts(TradeLog(trades))
        # TagVerdict 已刻意不含 level / is_significant(避免多重比較偽陽性)
        for t in tv:
            assert not hasattr(t, "level"), "per-tag 不應再有裁決等級"
            assert not hasattr(t, "is_significant"), "per-tag 不應再做顯著性宣稱"


def test_per_tag_still_descriptive_and_sorted():
    """降級為描述統計後,仍要能揪出『哪一招在賠錢』且最差排最前。"""
    trades = [_t(-100, i, tag="壞招") for i in range(10)]
    trades += [_t(100, i + 20, tag="好招") for i in range(10)]
    tv = per_tag_verdicts(TradeLog(trades))
    assert tv[0].tag == "壞招"
    assert tv[0].is_losing and not tv[1].is_losing
    assert "賠錢" in tv[0].descriptor and "賺錢" in tv[1].descriptor


# ── P1: 反事實不得宣稱「你其實是有救的」 ───────────────────
def test_counterfactual_no_salvation_claim():
    trades = [_t(-200, i, tag="壞招") for i in range(15)]
    trades += [_t(100, i + 30, tag="好招") for i in range(15)]
    r = analyze_log(TradeLog(trades), n_bootstrap=600)
    cf = r.counterfactual
    assert cf is not None
    assert "有救" not in cf.message, "不可下『不做這一招你就有救』的認證結論"
    assert "回歸均值" in cf.message, "必須誠實揭露事後挑選的偏誤"


# ── P1: required_sample_size 納入 power,不再低估 ───────────
def test_required_sample_size_includes_power():
    import math
    wr, po = 0.55, 1.2
    new = required_sample_size(wr, po)
    edge = wr * po - (1 - wr)
    var = wr * (po - edge) ** 2 + (1 - wr) * (-1 - edge) ** 2
    old = max(30, math.ceil((2 * math.sqrt(var) / edge) ** 2))
    assert new > old, "納入 power 項後估計應更保守(更大)"
    assert abs((Z_ALPHA_ONE_SIDED + Z_POWER_80) - 2.4865) < 0.01


def test_required_sample_size_from_pnls():
    rng = random.Random(1)
    pnls = [rng.gauss(8, 60) for _ in range(80)]
    n = required_sample_size_from_pnls(pnls)
    assert n is not None and n > 30
    # 負期望 → None(再多樣本也沒用)
    assert required_sample_size_from_pnls([-5.0] * 20) is None


# ── P2: T4 改名 + deprecated alias ─────────────────────────
def test_holdout_validate_renamed_with_alias():
    from core.backtest import holdout_validate, walk_forward_validate
    log = TradeLog([_t(10, i) for i in range(30)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        walk_forward_validate(log, n_bootstrap=200)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
    # 新名字正常運作
    assert holdout_validate(log, n_bootstrap=200) is not None


# ── P2: T5 edge_margin 警訊 ────────────────────────────────
def test_thin_edge_margin_flag():
    """勝率 70% + 盈虧比 0.5(打平門檻 0.43)→ 應觸發安全邊際薄警訊。"""
    pnls = [50] * 28 + [-100] * 12
    v = judge(TradeLog([_t(p, i) for i, p in enumerate(pnls)]), n_bootstrap=500)
    assert v.metrics.expectancy > 0            # 正期望(硬切點抓不到)
    assert any(f.code == "thin_edge_margin" for f in v.red_flags)


# ── P2: T6 opt-in 放空 ─────────────────────────────────────
def test_short_default_rejected():
    b = PaperBroker(cash=100_000)
    b.set_price("X", 100)
    r = b.place_order(Order("X", OrderSide.SELL, 50))
    assert not r.ok
    assert b.get_account().cash == 100_000


def test_short_opt_in_pnl_correct_with_disclaimer():
    b = PaperBroker(cash=100_000, fee_rate=0, slippage=0, allow_short=True)
    b.set_price("X", 100)
    r1 = b.place_order(Order("X", OrderSide.SELL, 10))   # 開空 @100
    assert r1.ok
    assert "借券費" in r1.message                        # 免責必須出現
    assert b.get_positions()[0].quantity == -10
    b.set_price("X", 80)
    b.place_order(Order("X", OrderSide.BUY, 10))         # 回補 @80
    assert abs((b.get_account().cash - 100_000) - 200) < 1e-6
    assert len(b.get_positions()) == 0


# ── as_dict 補齊 counterfactual / breakeven ────────────────
def test_as_dict_has_counterfactual_and_breakeven():
    import json
    trades = [_t(-200, i, tag="壞招") for i in range(15)]
    trades += [_t(100, i + 30, tag="好招") for i in range(15)]
    d = analyze_log(TradeLog(trades), n_bootstrap=400).as_dict()
    json.dumps(d, ensure_ascii=False)   # 必須可序列化
    assert d["counterfactual"] is not None
    assert "breakeven" in d
    # per-tag JSON 不得含優勢認證欄位
    for tv in d["tag_verdicts"]:
        assert "level" not in tv and "is_significant" not in tv


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
