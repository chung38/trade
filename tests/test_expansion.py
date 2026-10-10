"""新模組測試:市場擴充、倖存者偏差、假老師驗證、話術偵測、鑑識、風險模擬、HTML。

每個數學主張都對照解析解或已知值,防止未來改壞。
執行: python tests/test_expansion.py
"""

from __future__ import annotations

import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyzer import analyze_log  # noqa: E402
from core.antiscam.guru_claim import (  # noqa: E402
    analyze_guru_claim,
    binomial_tail_ge,
    years_to_own_the_world,
)
from core.antiscam.text_scanner import scan_text  # noqa: E402
from core.forensics import analyze_returns  # noqa: E402
from core.forensics.stat_tests import chi2_sf, runs_test, sharpe_with_ci  # noqa: E402
from core.ingest.costs import estimate_round_trip_cost  # noqa: E402
from core.markets import contract_multiplier, infer_market  # noqa: E402
from core.models import Market, Side, Trade, TradeLog  # noqa: E402
from core.montecarlo import (  # noqa: E402
    gambler_ruin_probability,
    losing_streak_probability,
    simulate_ruin_scenario,
)
from core.report_html import render_html_report, render_share_card  # noqa: E402
from core.survivorship import (  # noqa: E402
    guru_illusion,
    prob_perfect_run,
    prob_streak_in_trials,
)


def _t(pnl, i, tag=None):
    d = datetime(2024, 1, 1) + timedelta(days=i)
    return Trade("X", Market.US_STOCK, Side.LONG, d, d + timedelta(days=1),
                 100, 100, 10, fees=0, pnl=pnl, tag=tag)


# ══════ S4: 市場與契約乘數 ══════
def test_forex_not_misclassified_as_crypto():
    """回歸測試:舊版因『以 USD 結尾』把外匯誤判為加密貨幣。"""
    assert infer_market("EURUSD") == Market.FOREX
    assert infer_market("GBPUSD") == Market.FOREX
    assert infer_market("AUDUSD") == Market.FOREX
    assert infer_market("BTCUSDT") == Market.CRYPTO


def test_market_classification():
    assert infer_market("2330") == Market.TW_STOCK
    assert infer_market("0050") == Market.TW_ETF
    # 期貨/選擇權必須帶契約月份碼
    assert infer_market("TXFG5") == Market.TW_FUTURES
    assert infer_market("TXF202607") == Market.TW_FUTURES
    assert infer_market("TXO18000G5") == Market.TW_OPTIONS
    assert infer_market("AAPL") == Market.US_STOCK
    assert infer_market("ZZZZZZ") == Market.UNKNOWN   # 模稜兩可 → 不猜


def test_contract_multiplier_whitelist():
    assert contract_multiplier("TXF202607") == (200.0, True)
    assert contract_multiplier("MXFH5") == (50.0, True)
    # 查不到就是查不到,不猜
    assert contract_multiplier("2330") == (1.0, False)


def test_real_us_tickers_not_hijacked_by_futures_prefix():
    """回歸測試(gpt-5.5/grok-4.5 辯論發現):TMF/FXF/TXO/EXF 是真實美股代號,
    裸前綴比對會把美股使用者的損益放大 10~4000 倍。"""
    for s in ("TMF", "FXF", "TXO", "EXF"):
        assert infer_market(s) == Market.US_STOCK, s
        assert contract_multiplier(s) == (1.0, False), s


def test_futures_pnl_uses_multiplier():
    """台指期漲 100 點 × 1 口 = 20,000 元(不是 100 元)。"""
    d = datetime(2025, 1, 2)
    t = Trade("TXF", Market.TW_FUTURES, Side.LONG, d, d, 23000, 23100, 1,
              fees=0, contract_multiplier=200.0)
    assert t.pnl == 20000


def test_stock_backward_compatible():
    """乘數預設 1.0,股票結果與舊版逐位相同。"""
    d = datetime(2025, 1, 2)
    t = Trade("2330", Market.TW_STOCK, Side.LONG, d, d, 1000, 1080, 1000, fees=100)
    assert t.pnl == 79900
    assert t.contract_multiplier == 1.0


def test_futures_cost_both_sides_tax():
    """期交稅買賣雙邊都收,且課契約價值。"""
    c = estimate_round_trip_cost(Market.TW_FUTURES, Side.LONG, 23000, 23100, 1,
                                 contract_multiplier=200.0)
    # 手續費 20×2 + 稅 0.00002×契約價值×2 + 滑價
    assert 500 < c < 900   # 實務上台指期一口來回約 500~800 元


# ══════ A2: 倖存者偏差(解析解)══════
def test_streak_dp_matches_closed_form():
    """T = N 時,連勝機率精確等於 p^N。"""
    for n in (3, 5, 10):
        assert abs(prob_streak_in_trials(n, n, 0.5) - 0.5 ** n) < 1e-12


def test_streak_dp_matches_monte_carlo():
    def mc(T, N, p, trials=40000, seed=1):
        rng = random.Random(seed)
        hit = 0
        for _ in range(trials):
            run = 0
            for _ in range(T):
                if rng.random() < p:
                    run += 1
                    if run >= N:
                        hit += 1
                        break
                else:
                    run = 0
        return hit / trials

    for T, N, p in [(20, 5, 0.5), (30, 4, 0.6)]:
        assert abs(prob_streak_in_trials(T, N, p) - mc(T, N, p)) < 0.01


def test_guru_illusion_is_probabilistic_not_certain():
    """絕不能用『必然存在』的語氣 —— 實際機率不是 1。"""
    g = guru_illusion(n_traders=1000, n_trials=10, streak=10)
    assert 0.6 < g.prob_at_least_one < 0.65     # 約 0.6236
    assert g.prob_at_least_one < 1.0
    assert "一定" not in g.headline and "必然" not in g.headline


def test_perfect_run_probability():
    assert abs(prob_perfect_run(10, 0.5) - 1 / 1024) < 1e-12


# ══════ A4: 假老師驗證器 ══════
def test_binomial_tail_matches_hand_calc():
    exact = (math.comb(10, 8) + math.comb(10, 9) + math.comb(10, 10)) / 1024
    assert abs(binomial_tail_ge(10, 8, 0.5) - exact) < 1e-12
    assert binomial_tail_ge(10, 0, 0.5) == 1.0
    assert binomial_tail_ge(10, 11, 0.5) == 0.0


def test_compounding_reductio():
    y = years_to_own_the_world(0.20)
    assert y is not None and 10 < y < 13
    assert years_to_own_the_world(-0.05) is None    # 負報酬不發散


def test_guru_claim_detects_contradiction():
    """宣稱勝率 60% + 盈虧比 0.5 → 期望值為負,自相矛盾。"""
    a = analyze_guru_claim(claimed_win_rate=0.6, payoff_ratio=0.5)
    assert a.verdict == "宣稱自相矛盾"


def test_guru_claim_refuses_without_sample_size():
    """只說勝率不說筆數 → 誠實說「無法檢驗」,不編數字。"""
    a = analyze_guru_claim(claimed_win_rate=0.9)
    assert a.verdict == "資訊不足,無法檢驗"
    assert a.luck_prob_win_rate is None


# ══════ A3: 話術偵測(誤報率是關鍵)══════
def test_scanner_catches_scams():
    scams = [
        "老師帶單超準!加入VIP群保證獲利,名額有限,私訊我加客服",
        "你的獲利要出金需先繳稅,請先匯款到平台帳戶刷流水解凍",
    ]
    for s in scams:
        assert scan_text(s).risk_level in ("高", "極高")


def test_scanner_no_false_positive_on_normal_discussion():
    normals = [
        "我覺得台積電基本面不錯,但現在本益比偏高,想等回檔再進場",
        "定期定額買 0050 是不錯的長期策略,不用擇時",
        "最近大盤震盪,我把部位降到五成,設好停損",
    ]
    for s in normals:
        assert scan_text(s).risk_level not in ("高", "極高"), s


def test_scanner_does_not_flag_anti_scam_education():
    """反詐教育文含關鍵字但在否定語境,不可誤判(雙向否定視窗)。"""
    edus = [
        "任何說保證獲利的都是騙人的,千萬不要相信",
        "老師帶單、升級VIP,這些都是詐騙的典型手法,請小心",
        "提醒大家:出金要繳稅的說法 100% 是詐騙,請撥 165",
    ]
    for s in edus:
        assert scan_text(s).risk_level not in ("高", "極高"), s


def test_scanner_resists_self_denial_evasion():
    """詐騙犯自稱『我們不是詐騙』不該讓風險降級,反而是警訊。"""
    r = scan_text("我們絕不是詐騙!保證獲利穩賺不賠,快加客服升級VIP")
    assert r.risk_level in ("高", "極高")
    assert any(h.category == "self_denial" for h in r.hits)


# ══════ A1: 鑑識 ══════
def test_chi2_sf_matches_table():
    for x, df, exp in [(3.841, 1, 0.05), (16.919, 9, 0.05), (21.666, 9, 0.01)]:
        assert abs(chi2_sf(x, df) - exp) < 0.001


def test_runs_test_type_i_error_calibrated():
    hits = 0
    N = 800
    for s in range(N):
        rng = random.Random(s)
        r = runs_test([rng.gauss(0, 1) for _ in range(40)])
        if r and r.p_two_sided < 0.05:
            hits += 1
    assert 0.02 < hits / N < 0.08


def test_sharpe_lo_standard_error():
    rng = random.Random(3)
    rets = [rng.gauss(0.02, 0.03) for _ in range(36)]
    s = sharpe_with_ci(rets, periods_per_year=12)
    m = sum(rets) / 36
    var = sum((x - m) ** 2 for x in rets) / 35
    sr = m / math.sqrt(var)
    se = math.sqrt((1 + 0.5 * sr * sr) / 36) * math.sqrt(12)
    assert abs(s.std_error - se) < 1e-9


def test_forensics_catches_madoff_like_returns():
    rng = random.Random(7)
    smooth = [abs(rng.gauss(0.0095, 0.002)) for _ in range(60)]
    f = analyze_returns(smooth)
    assert f.suspicion_level in ("高度可疑", "可疑")
    assert any(x.code == "almost_never_loses" for x in f.findings)


def test_forensics_does_not_flag_legit_strategy():
    rng = random.Random(11)
    legit = [rng.gauss(0.008, 0.04) for _ in range(48)]
    f = analyze_returns(legit)
    assert f.suspicion_level in ("未發現明顯疑點", "略有疑點")


def test_forensics_never_convicts():
    """每個 finding 都必須說明「這不代表什麼」—— 絕不自動定罪造假。"""
    rng = random.Random(7)
    f = analyze_returns([abs(rng.gauss(0.0095, 0.002)) for _ in range(60)])
    for x in f.findings:
        assert x.does_not_mean, f"{x.code} 缺少『不代表什麼』的誠實揭露"
        assert "造假" not in x.what   # what 只描述觀察,不定罪


# ══════ S1: 風險模擬 ══════
def test_gambler_ruin_closed_form():
    assert abs(gambler_ruin_probability(5, 10, 0.5) - 0.5) < 1e-12
    assert abs(gambler_ruin_probability(3, 10, 0.5) - 0.7) < 1e-12
    # p != 0.5 對照解析式
    r = (1 - 0.45) / 0.45
    exp = (r ** 5 - r ** 10) / (1 - r ** 10)
    assert abs(gambler_ruin_probability(5, 10, 0.45) - exp) < 1e-12


def test_gambler_ruin_matches_simulation():
    def mc(a, N, p, paths=20000, seed=1):
        rng = random.Random(seed)
        ruin = 0
        for _ in range(paths):
            x = a
            while 0 < x < N:
                x += 1 if rng.random() < p else -1
            if x <= 0:
                ruin += 1
        return ruin / paths
    assert abs(gambler_ruin_probability(5, 10, 0.45) - mc(5, 10, 0.45)) < 0.02


def test_drawdown_never_exceeds_100pct():
    """爆倉是吸收狀態 —— 不可能出現 181% 的回撤(回歸測試)。"""
    rng = random.Random(2)
    s = simulate_ruin_scenario([rng.gauss(-20, 100) for _ in range(40)],
                               start_equity=10000, n_paths=800)
    assert s is not None
    assert s.p95_max_drawdown <= 1.0
    assert s.median_max_drawdown <= 1.0


def test_ruin_sim_discriminates():
    rng = random.Random(2)
    bad = simulate_ruin_scenario([rng.gauss(-20, 100) for _ in range(40)],
                                 start_equity=10000, n_paths=800)
    good = simulate_ruin_scenario([rng.gauss(30, 80) for _ in range(40)],
                                  start_equity=10000, n_paths=800)
    assert bad.ruin_fraction > good.ruin_fraction


def test_ruin_sim_warns_about_tail_underestimation():
    rng = random.Random(2)
    s = simulate_ruin_scenario([rng.gauss(10, 50) for _ in range(30)], n_paths=400)
    assert any("尾端低估" in w for w in s.warnings)


def test_ruin_sim_refuses_small_sample():
    assert simulate_ruin_scenario([1.0] * 5) is None


def test_losing_streak_probability():
    # 勝率越低,連虧越容易
    assert (losing_streak_probability(200, 10, 0.4)
            > losing_streak_probability(200, 10, 0.7))


# ══════ S2: HTML 報告 ══════
def _sample_result():
    trades = [_t(-50, i, "壞招") for i in range(20)]
    trades += [_t(30, i + 30, "好招") for i in range(20)]
    return analyze_log(TradeLog(trades), n_bootstrap=300)


def test_html_report_escapes_xss():
    trades = [_t(-50, i, '</script><img src=x onerror=alert(1)>') for i in range(20)]
    trades += [_t(30, i + 30, "正常") for i in range(20)]
    html = render_html_report(analyze_log(TradeLog(trades), n_bootstrap=300))
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img" in html


def test_html_report_no_future_projection():
    html = render_html_report(_sample_result())
    for banned in ("你將會有", "預期未來", "預測未來報酬"):
        assert banned not in html


def test_share_card_renders():
    card = render_share_card(_sample_result())
    assert "不構成投資建議" in card
    assert "目前適合的階段" in card
    assert "USD（推定）" in card
    assert len(card) > 500


# ══════ 最強模型辯論(gpt-5.5 + grok-4.5 + opus4.8)修正的防退化測試 ══════
def test_binomial_large_n_no_overflow():
    """舊版 math.comb 逐項累加在 n=10000 直接 OverflowError 崩潰。"""
    r = binomial_tail_ge(10000, 5500, 0.5)
    assert 0.0 <= r <= 1e-20
    r2 = binomial_tail_ge(1000000, 505000, 0.5)   # 一百萬筆也要能算
    assert 0.0 <= r2 <= 1.0


def test_drawdown_capital_base_uses_multiplier():
    """回撤資本基準必須含契約乘數,否則期貨的回撤 % 錯 200 倍。"""
    from core.metrics.performance import compute_metrics
    d = datetime(2025, 1, 2)
    trades = [
        Trade("TXFG5", Market.TW_FUTURES, Side.LONG, d, d + timedelta(days=i),
              23000, 23000 + (50 if i % 2 else -50), 1,
              fees=0, contract_multiplier=200.0)
        for i in range(12)
    ]
    m = compute_metrics(TradeLog(trades))
    # 資本基準 = 23000×200 = 4.6M;單筆虧 1 萬,回撤 % 必須遠小於 1%
    assert m.max_drawdown_pct < 0.05


def test_losing_streak_ignores_breakeven():
    """打平交易不是虧損:連虧機率不得把 pnl==0 算進虧損率。"""
    rng = random.Random(5)
    # 一半打平、少量虧損:真實虧損率低,連虧 10 次機率應極小
    pnls = [0.0] * 20 + [50.0] * 15 + [-30.0] * 5
    s = simulate_ruin_scenario(pnls, start_equity=100000, n_paths=300)
    # 虧損率 = 5/40 = 12.5% → 連虧10次機率 ≈ (0.125)^10 量級,幾乎為 0
    assert s.losing_streak_10_prob < 0.001


def test_scanner_fullwidth_and_zerowidth_evasion():
    """NFKC 正規化 + 零寬字元剝除:全形/夾字規避必須被抓到。"""
    r1 = scan_text("老師帶單保證獲利,快加客服升級ＶＩＰ,名額有限")
    assert r1.risk_level in ("高", "極高")
    r2 = scan_text("老師帶單,保​證​獲​利穩賺不賠,快加客服")
    assert r2.risk_level in ("高", "極高")


def test_scanner_first_negated_occurrence_not_terminal():
    """首次出現被否定不可就此打住 —— 後面的真話術要抓到。"""
    r = scan_text("有人說保證獲利都是騙人的。但我們不一樣!真的保證獲利穩賺不賠,快加客服升級VIP")
    assert r.risk_level in ("高", "極高")


def test_scanner_discussion_gate_not_exploitable():
    """詐騙文夾一句求證語,不得觸發討論降級(相異類別 >= 3 時不降)。"""
    r = scan_text("請問這是詐騙嗎?開玩笑的!老師帶單保證獲利,升級VIP名額有限,快匯款到平台入金")
    assert r.risk_level in ("高", "極高")


def test_forensics_zero_volatility_finding():
    """恆定報酬(最露骨的龐氏樣態)不得因 runs/sharpe 回 None 而隱形。"""
    f = analyze_returns([0.01] * 24)
    assert any(x.code == "zero_volatility" for x in f.findings)
    assert f.suspicion_level in ("高度可疑", "可疑")


def test_forensics_survives_nan_inf():
    """NaN / inf 不得汙染統計或造成崩潰。"""
    rng = random.Random(3)
    vals = [rng.gauss(0.01, 0.03) for _ in range(30)] + [float("nan"), float("inf")]
    f = analyze_returns(vals)
    assert f.n_periods == 30                     # 非有限值被剔除
    assert math.isfinite(f.mean_return) and math.isfinite(f.volatility)


def test_montecarlo_inferred_equity_flagged():
    rng = random.Random(4)
    pnls = [rng.gauss(10, 50) for _ in range(30)]
    s = simulate_ruin_scenario(pnls, n_paths=200)          # 未給權益 → 推估
    assert s.start_equity_inferred
    assert any("粗估" in w for w in s.warnings)
    s2 = simulate_ruin_scenario(pnls, start_equity=100000, n_paths=200)
    assert not s2.start_equity_inferred


def test_montecarlo_param_validation():
    pnls = [1.0] * 20
    for bad_kwargs in (
        {"n_paths": 0}, {"ruin_drawdown": 1.5}, {"start_equity": -5},
    ):
        try:
            simulate_ruin_scenario(pnls, **bad_kwargs)
            assert False, f"應拒絕 {bad_kwargs}"
        except ValueError:
            pass


def test_loader_rejects_mixed_timezone_bases():
    """有 offset 與 naive 掛鐘時間混用時，不能假裝在同一時間基準排序。"""
    import tempfile
    from core.ingest.loader import load_trades
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "tz.csv"
        f.write_text(
            "代號,方向,進場時間,出場時間,進場價,出場價,數量\n"
            "2330,買,2025-01-03T09:00:00+08:00,2025-01-03T13:00:00+08:00,1000,1010,1000\n"
            "2330,買,2025-01-06,2025-01-07,1000,1020,1000\n",
            encoding="utf-8-sig",
        )
        try:
            load_trades(f, market_hint=Market.TW_STOCK)
        except ValueError as exc:
            assert "混用了有時區與無時區" in str(exc)
        else:
            raise AssertionError("時間基準不一致時不得排序或做 OOS")


# ══════ 第四輪巡檢(gpt-5.5 + grok-4.5 + opus4.8)修正的防退化測試 ══════
def test_scanner_simplified_chinese_detected():
    """簡體詐騙文原本 0 命中判「低」—— 詐騙集團常用簡體發文,不可有此盲區。"""
    r = scan_text("老师带单保证获利,稳赚不赔,快升级VIP,名额有限,私讯我")
    assert r.risk_level in ("高", "極高")
    # 簡體反詐教育文的警示詞也要能觸發否定
    r2 = scan_text("任何说保证获利的都是骗人的,千万不要相信,这是诈骗")
    assert r2.risk_level not in ("高", "極高")
    # 繁體行為完全不變(translate 對繁體是恆等)
    r3 = scan_text("定期定額買 0050 是不錯的長期策略,不用擇時")
    assert r3.risk_level not in ("高", "極高")


def test_bootstrap_validates_and_caps():
    """n_bootstrap < 1 要拋 ValueError(否則除零/空索引);大樣本自動降次數。"""
    from core.verdict.statistics import MAX_BOOTSTRAP_DRAWS, test_expectancy_positive
    try:
        test_expectancy_positive([1.0, 2.0, 3.0], n_bootstrap=0)
        assert False, "應拋 ValueError"
    except ValueError:
        pass
    # 大樣本:20M 上限生效,不會跑滿 5000 次(用耗時間接驗證:< 25s 的舊行為)
    import time
    big = [float(i % 7 - 3) for i in range(10000)]
    t0 = time.time()
    test_expectancy_positive(big, n_bootstrap=5000)
    assert time.time() - t0 < 10, "bootstrap 上限未生效(大樣本過慢)"
    assert MAX_BOOTSTRAP_DRAWS == 20_000_000


def test_examples_packaged_inside_core():
    """examples 必須在 core/ 套件內(否則 pip 安裝後 demo 全壞)。"""
    from core.cli import _examples_dir
    d = _examples_dir()
    assert d.name == "examples" and d.parent.name == "core"
    assert (d / "tw_stock_gambling.csv").exists()


def test_forensics_n_dropped_surfaced():
    """NaN 剔除數必須揭露在 as_dict 與文字報告(不可靜默)。"""
    from core.forensics import render_forensics
    vals = [0.01] * 20 + [float("nan")] * 3
    f = analyze_returns(vals)
    assert f.n_dropped == 3
    assert f.as_dict()["n_dropped"] == 3
    assert "剔除" in render_forensics(f)


def test_chart_cdn_pinned():
    """CDN 必須鎖版:未鎖版的 lightweight-charts 已被 v5 破壞(移除 v4 API)。"""
    from core.charts.registry import get_chart_lib
    from core.charts import preview
    lw = get_chart_lib("lightweight").module_code
    assert "lightweight-charts@4" in lw, "lightweight 未鎖 v4"
    assert "lightweight-charts@4" in preview._PREVIEW_TEMPLATE
    ec = get_chart_lib("echarts").module_code
    assert "echarts@5" in ec, "echarts 未鎖版"


def test_mplfinance_template_markers_and_no_none_addplot():
    """mplfinance 範本:標記要真的填值;無訊號時不可傳 addplot=None(會 TypeError)。"""
    src = None
    from core.charts.registry import get_chart_lib
    src = get_chart_lib("mplfinance").module_code
    assert "addplot=addplots or None" not in src   # 舊寫法會讓無訊號的專案崩潰
    assert "pos_of" in src and 'buy_y[i] = m["price"]' in src   # 標記真的被填
    import ast
    ast.parse(src)


def test_cli_stdin_utf8_no_silent_miss():
    """Windows cp950 stdin 管線不得靜默漏抓詐騙(反詐工具最諷刺的失敗)。"""
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    r = subprocess.run(
        [sys.executable, "-m", "core.cli", "scan-text"],
        input="老師帶單保證獲利,穩賺不賠,快升級VIP".encode("utf-8"),
        capture_output=True, env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert r.returncode == 2, "UTF-8 stdin 的詐騙文應判高風險(exit 2)"


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
