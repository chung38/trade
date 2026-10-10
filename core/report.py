"""把分析結果彙整成一份完整、誠實、好讀的中文報告。"""

from __future__ import annotations

from .antiscam.signals import scam_warnings_for
from .backtest.validate import OutOfSampleReport
from .metrics.breakeven import compute_break_even
from .metrics.performance import PerformanceMetrics, fmt_ratio
from .markets import uncovered_cost_warnings
from .models import TradeLog
from .onboarding import stage_from_components
from .strategy.profiler import StrategyProfile
from .verdict.judge import Verdict, VerdictLevel


def _money(x: float) -> str:
    return f"{x:,.2f}"


def render_text_report(
    log: TradeLog,
    metrics: PerformanceMetrics,
    verdict: Verdict,
    profile: StrategyProfile,
    oos: OutOfSampleReport | None = None,
    *,
    tag_verdicts=None,
    counterfactual=None,
    follow_guru=None,
) -> str:
    """產生純文字報告(適合終端機輸出)。"""
    L: list[str] = []
    L.append("=" * 70)
    L.append("                  反詐投資王 — 交易績效誠實報告")
    L.append("=" * 70)
    L.append(f"資料來源: {log.source}")
    L.append(f"涵蓋市場: {', '.join(sorted(m.value for m in log.markets))}")
    L.append("")

    # ── 一句話裁決 ──
    L.append("【最終裁決】")
    L.append(f"  {verdict.headline}")
    L.append("")

    # 新手最先需要的不是術語，而是「現在到底能做哪一步」。這個分流比
    # 裁決更保守：樣本內通過但樣本外未延續，仍只建議紙上模擬。
    stage = stage_from_components(verdict=verdict, metrics=metrics, oos=oos)
    L.append("【目前適合的階段】")
    L.append(f"  {stage.title}")
    L.append(f"  原因: {stage.reason}")
    L.append(f"  先做: {stage.next_actions[0]}")
    L.append("")

    # ── 核心數字 ──
    m = metrics
    currency = f" {m.pnl_currency}" if m.pnl_currency else ""
    L.append("【核心績效】")
    L.append(f"  交易筆數      : {m.total_trades}(勝 {m.wins} / 負 {m.losses})")
    L.append(f"  勝率          : {m.win_rate:.1%}(含打平;打平 {m.breakeven} 筆)")
    L.append(f"  盈虧比        : {fmt_ratio(m.payoff_ratio)}(平均賺 {_money(m.avg_win)} / 平均賠 {_money(m.avg_loss)})")
    L.append(f"  獲利因子      : {fmt_ratio(m.profit_factor)}")
    L.append(f"  每筆期望值    : {_money(m.expectancy)}{currency}  ← 最關鍵的單一數字")
    L.append(f"  總損益        : {_money(m.total_pnl)}{currency}")
    L.append(
        f"  費用揭露      : {_money(m.total_fees)}{currency}"
        "（直接 pnl 須已扣成本；此欄不會重複扣除）"
    )
    if m.sequence_metrics_reliable:
        dd_pct_txt = (
            f"{m.max_drawdown_pct:.1%}" if m.drawdown_pct_reliable
            else "%無法計算 — 缺可靠資本/帳戶權益基準"
        )
        L.append(f"  最大回撤      : {_money(m.max_drawdown)}{currency}({dd_pct_txt})")
        L.append(f"  最長連虧      : {m.max_consecutive_losses} 次")
    else:
        L.append("  最大回撤      : 無法計算（出場先後順序不可靠）")
        L.append("  最長連虧      : 無法計算（出場先後順序不可靠）")
    if m.return_metrics_reliable:
        L.append(f"  夏普 / 索提諾 : {fmt_ratio(m.sharpe)} / {fmt_ratio(m.sortino)}(每筆基準,非年化)")
    else:
        L.append("  夏普 / 索提諾 : 無法計算（缺可信價量或契約乘數）")
    L.append(f"  單筆最大賺/賠 : {_money(m.largest_win)} / {_money(m.largest_loss)}")
    # 「最賺一筆佔比」只在有 2 筆以上獲利時才有意義(僅 1 筆時必為 100%,是噪音)
    if m.wins > 1:
        L.append(f"  最賺一筆佔比  : {m.top_trade_pnl_share:.1%} 的總獲利")
    if m.sequence_note:
        L.append(f"  ⚠ {m.sequence_note}")
    if m.return_note:
        L.append(f"  ⚠ {m.return_note}")
    if getattr(m, "drawdown_note", ""):
        L.append(f"  ⚠ {m.drawdown_note}")
    if getattr(m, "currency_note", ""):
        L.append(f"  ⚠ {m.currency_note}")
    L.append("")

    # ── 統計顯著性 ──
    sig = verdict.significance
    L.append("【這是優勢,還是運氣?(統計檢定)】")
    L.append(f"  每筆平均損益          : {_money(sig.mean)}{currency}")
    L.append(
        f"  95% 信賴區間          : [{_money(sig.ci_low)}, {_money(sig.ci_high)}]{currency}"
    )
    L.append(f"  t 檢定 p 值           : {sig.p_value_t:.4f}")
    L.append(f"  Bootstrap p 值        : {sig.p_value_bootstrap:.4f}")
    verdict_word = "顯著為正(像真優勢)" if sig.is_significant else "不顯著(無法排除是運氣)"
    L.append(f"  結論                  : {verdict_word}")
    # 負期望時,「需要多少樣本」沒有意義(再多樣本也無法把負期望變成優勢),
    # 不應把內部哨兵值(如 9999)直接印給使用者。
    if m.expectancy <= 0:
        L.append(
            "  建議最少交易筆數      : 不適用 — 期望值為負,"
            "問題不在筆數,而在方法;再多交易也無法變成優勢"
        )
    else:
        L.append(f"  建議最少交易筆數      : {verdict.required_trades}(目前 {m.total_trades})")
    L.append("")

    # ── 賭博警訊 ──
    if verdict.red_flags:
        L.append("【偵測到的賭博 / 風險警訊】")
        for rf in verdict.red_flags:
            tag = {"high": "🔴 高", "medium": "🟠 中", "low": "🟡 低"}.get(rf.severity, "•")
            L.append(f"  {tag} {rf.message}")
        L.append("")

    # ── 樣本外驗證 ──
    if oos is not None:
        L.append("【樣本外驗證(揭穿過度配適 / 倖存者偏差)】")
        L.append(f"  {oos.headline}")
        if getattr(oos, "available", True):
            L.append(
                f"  樣本內: {oos.in_sample.n_trades} 筆, 期望值 {_money(oos.in_sample.expectancy)}{currency}"
            )
            L.append(
                f"  樣本外: {oos.out_sample.n_trades} 筆, 期望值 {_money(oos.out_sample.expectancy)}{currency}"
            )
        else:
            L.append(f"  無法切分: {oos.unavailable_reason}")
        for line in oos.interpretation:
            L.append(f"    - {line}")
        L.append("")

    # ── 策略輪廓 ──
    L.append("【你的交易模式(反推)】")
    L.append(f"  風格          : {profile.style}")
    if profile.timing_metrics_available:
        L.append(f"  平均持倉      : {profile.avg_holding_days:.1f} 天")
    else:
        L.append("  平均持倉      : 無法計算（缺少可靠的進出場時間）")
    L.append(f"  標的集中度    : 前三大標的佔 {profile.symbol_concentration:.0%}(共 {profile.distinct_symbols} 檔)")
    for note in profile.notes:
        L.append(f"  ⚠ {note}")
    L.append("")

    # ── 逐策略體檢(描述統計,不做優勢認證)──
    if tag_verdicts:
        L.append("【🔍 各策略體檢 — 哪一招在送錢?(由最差到最好)】")
        L.append("  策略標籤            筆數   每筆期望值      總損益        狀態")
        L.append("  " + "─" * 66)
        for tv in tag_verdicts:
            L.append(
                f"  {tv.tag[:16]:<16}  {tv.n_trades:>4}   "
                f"{_money(tv.expectancy):>12}  {_money(tv.total_pnl):>12}   "
                f"{tv.descriptor}"
            )
        L.append(
            "  註:這裡只呈現『描述統計』,不對個別策略做優勢認證 —— 因為對多個策略"
        )
        L.append(
            "      各做一次統計檢定,會讓運氣被誤認成優勢(策略越多、誤判機率越高)。"
        )
        L.append("")

    # ── 跟單 / 聽明牌的成績單(反詐實用化:用你自己的數字檢驗跟單績效)──
    if follow_guru is not None:
        L.append("【🎯 跟單 / 聽明牌的成績單】")
        L.append(f"  {follow_guru.message}")
        if follow_guru.follow_tags:
            L.append(f"  (涵蓋標籤:{', '.join(follow_guru.follow_tags[:5])})")
        L.append("")

    # ── 反事實:停掉最差策略後會如何 ──
    if counterfactual is not None:
        L.append("【💡 如果停掉最差的一招】")
        L.append(f"  {counterfactual.message}")
        L.append("")

    # ── 離轉正還差多少(期望值為負時,給具體可行動的目標數字)──
    if metrics.expectancy <= 0 and metrics.total_trades > 0:
        targets = compute_break_even(metrics)
        if targets.messages and not targets.already_positive:
            L.append("【📐 離轉正還差多少 — 具體目標】")
            for msg in targets.messages:
                L.append(f"  • {msg}")
            L.append("")

    # ── 建議 ──
    # 未涵蓋成本 / 模型限制:markets.py 定義的警語必須真的到使用者眼前,
    # 不能只寫在程式碼註解裡(外匯 swap、永續資金費率、選擇權左尾)。
    cost_notes: list[str] = []
    for _mkt in sorted({t.market for t in log.trades}, key=lambda m: m.value):
        for _w in uncovered_cost_warnings(_mkt):
            if _w not in cost_notes:
                cost_notes.append(_w)
    if cost_notes:
        L.append("【⚠ 本工具未涵蓋的成本 / 模型限制 — 誠實聲明】")
        for _w in cost_notes:
            L.append(f"  • {_w}")
        L.append("")

    L.append("【給你的建議】")
    for a in verdict.advice:
        L.append(f"  • {a}")
    L.append("")

    # ── 反詐警語(若分析結果命中詐騙受害特徵)──
    scam_warnings = scam_warnings_for(metrics, profile)
    if scam_warnings:
        L.append("【🛡 反詐提醒 — 你的交易可能與投資詐騙有關】")
        for w in scam_warnings:
            L.append(f"  {w}")
        L.append("  ── 想進一步檢測是否遇到詐騙,請執行:anti-gambling-trader scam-check")
        L.append("")

    # ── 勸退橫幅(若需要)──
    if verdict.should_discourage:
        L.append("┌" + "─" * 66 + "┐")
        L.append("│  ⛔ 勸退提醒                                                      │")
        L.append("│                                                                  │")
        if verdict.level == VerdictLevel.GAMBLING:
            L.append("│  根據統計分析,你目前的交易行為比較接近『賭博』而非投資。        │")
            L.append("│  樣本期望值為負:照這樣打下去,統計預期就是越虧越多。            │")
        elif verdict.level == VerdictLevel.LUCK_SUSPECTED:
            L.append("│  你帳面上賺錢,但統計上無法排除這只是運氣。                      │")
            L.append("│  別讓一時的好運,騙你以為自己找到了穩定獲利的方法。              │")
        elif verdict.level == VerdictLevel.INSUFFICIENT:
            L.append("│  你的交易次數還太少,任何結論(好或壞)都不可信。                │")
            L.append("│  在累積足夠樣本前,請勿放大部位、勿借錢、勿重押。                │")
        else:
            L.append("│  你的策略雖有統計訊號,但結構脆弱、風險偏高。                    │")
            L.append("│  請先修正警訊並通過樣本外驗證,再考慮自動化或加碼。              │")
        L.append("│                                                                  │")
        L.append("│  真正的投資優勢,經得起統計檢定與時間考驗。慢慢來,別賭。        │")
        L.append("└" + "─" * 66 + "┘")
    else:
        # 誠實性修正:裁決等級(judge)只看樣本內顯著性,「完全不看樣本外」。
        # 因此不能無條件宣稱「通過了樣本外驗證」—— 必須依實際的 oos 結果措辭,
        # 否則會出現「樣本外區塊寫『優勢消失』、結尾卻寫『通過樣本外驗證』」的自相矛盾。
        if oos is not None and oos.edge_persisted:
            L.append("✅ 你的策略通過了統計檢定,且樣本外驗證顯示優勢延續,屬於少數具備優勢的情況。")
        elif oos is not None:
            L.append("🟡 你的策略通過了統計檢定,但『樣本外驗證尚未確認』(見上方樣本外驗證區塊)。")
            L.append("   樣本內的優勢不等於未來的優勢 —— 請先確認優勢能延續,再考慮加碼或自動化。")
        else:
            L.append("✅ 你的策略通過了統計檢定(本次未做樣本外驗證)。")
        L.append("   但請記得:這是『目前』的證據,不是『未來』的保證。持續監控、嚴守紀律。")
    L.append("")
    L.append("─" * 70)
    L.append("免責聲明:本報告為統計分析工具的輸出,僅供教育與研究用途,")
    L.append("不構成任何投資建議。投資有風險,盈虧自負。過去績效不代表未來表現。")
    L.append("─" * 70)
    return "\n".join(L)
