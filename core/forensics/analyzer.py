"""假績效鑑識的高階編排。

輸入:老師/平台宣稱的報酬序列(如 36 個月的月報酬)。
輸出:一份「可疑徵兆」清單 —— 每一項都明講「這代表什麼、不代表什麼」。

⚠️ 本模組永不輸出「造假」「不可能」「確定是龐氏」這類定罪詞。
   它只說:「這個特徵值得你追問來源。」
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .stat_tests import (
    RunsTestResult,
    SharpeResult,
    TerminalDigitResult,
    lag1_autocorr,
    runs_test,
    sharpe_with_ci,
    terminal_digit_test,
)


@dataclass(frozen=True)
class Finding:
    """一個可疑徵兆。"""

    code: str
    severity: str          # "high" | "medium" | "low"(可疑程度,非定罪程度)
    what: str              # 觀察到什麼
    means: str             # 這代表什麼
    does_not_mean: str     # 這**不**代表什麼(誠實性關鍵)
    ask: str               # 你該追問什麼


@dataclass
class ReturnsForensics:
    """報酬序列的鑑識結果。"""

    n_periods: int
    positive_ratio: float
    mean_return: float
    volatility: float

    runs: RunsTestResult | None = None
    autocorr_lag1: float | None = None
    sharpe: SharpeResult | None = None
    terminal_digits: TerminalDigitResult | None = None
    round_number_ratio: float | None = None   # 純描述,不掛旗標

    findings: list[Finding] = field(default_factory=list)
    suspicion_level: str = "資訊不足"   # 序數,不給百分比
    n_dropped: int = 0                  # 被剔除的非數值期數(NaN/inf)——必須揭露,不可靜默

    def as_dict(self) -> dict:
        return {
            "n_periods": self.n_periods,
            "positive_ratio": self.positive_ratio,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "sharpe_annualized": self.sharpe.sharpe_annualized if self.sharpe else None,
            "sharpe_ci": (
                [self.sharpe.ci_low, self.sharpe.ci_high] if self.sharpe else None
            ),
            "runs_p": self.runs.p_two_sided if self.runs else None,
            "autocorr_lag1": self.autocorr_lag1,
            "terminal_digit_p": (
                self.terminal_digits.p_value if self.terminal_digits else None
            ),
            "round_number_ratio": self.round_number_ratio,
            "n_dropped": self.n_dropped,
            "suspicion_level": self.suspicion_level,
            "findings": [f.__dict__ for f in self.findings],
        }


ForensicsReport = ReturnsForensics   # 別名


def _round_number_ratio(values: list[float], *, decimals: int = 2) -> float:
    """尾數為 0 的比例。**純描述,不作為獨立旗標。**

    期望基準依報酬的尺度而定,沒有普適門檻 —— 硬訂一個門檻就是假精準。
    """
    if not values:
        return 0.0
    scale = 10 ** decimals
    zeros = sum(1 for v in values if int(abs(round(v * scale))) % 10 == 0)
    return zeros / len(values)


def analyze_returns(
    returns: list[float],
    *,
    periods_per_year: int = 12,
    label: str = "宣稱的報酬序列",
) -> ReturnsForensics:
    """對一段宣稱的報酬序列做統計鑑識。

    Args:
        returns:          報酬序列(0.02 = 2%)。至少要 8 期才做得了檢定。
        periods_per_year: 一年幾期(月報酬 = 12,週 = 52,日 = 252)
    """
    # 非有限值(NaN / inf)防護:NaN 會讓所有比較悄悄為 False、汙染均值與變異,
    # 產出看似正常實則錯誤的鑑識結果。直接剔除並記錄,不讓它汙染統計。
    clean = [r for r in returns if isinstance(r, (int, float)) and math.isfinite(r)]
    n_dropped = len(returns) - len(clean)
    returns = clean

    n = len(returns)
    if n == 0:
        # 「全是 NaN」是最極端的情況,更要揭露剔除數
        return ReturnsForensics(0, 0.0, 0.0, 0.0, suspicion_level="資訊不足",
                                n_dropped=n_dropped)

    mean = sum(returns) / n
    if n > 1:
        var = sum((r - mean) ** 2 for r in returns) / (n - 1)
        vol = var ** 0.5
    else:
        vol = 0.0
    pos_ratio = sum(1 for r in returns if r > 0) / n

    f = ReturnsForensics(
        n_periods=n,
        positive_ratio=pos_ratio,
        mean_return=mean,
        volatility=vol,
        runs=runs_test(returns),
        autocorr_lag1=lag1_autocorr(returns),
        sharpe=sharpe_with_ci(returns, periods_per_year=periods_per_year),
        # 尾數檢定用 decimals=4:報酬如 0.0095(=0.95%)在 decimals=2 下
        # 「最後一位」全是 0,檢定的是量級而非尾數偏好,毫無意義。
        terminal_digits=terminal_digit_test(returns, decimals=4),
        round_number_ratio=_round_number_ratio(returns, decimals=4),
        n_dropped=n_dropped,
    )

    findings: list[Finding] = []

    # ── 1. 連段太少:報酬被平滑化的徵兆 ──
    if f.runs and f.runs.too_few_runs and f.runs.p_two_sided < 0.05:
        findings.append(Finding(
            code="too_few_runs",
            severity="high",
            what=(
                f"正負號的連段只有 {f.runs.runs} 段,遠少於隨機排列的期望值 "
                f"{f.runs.expected_runs:.1f} 段(p={f.runs.p_two_sided:.4f})。"
            ),
            means="報酬的正負號高度聚集 —— 這是「報酬被人為平滑」的典型特徵之一。",
            does_not_mean=(
                "**不代表造假。** 趨勢性市場、動量策略、或持有到期的債券部位,"
                "都會自然產生連段偏少。"
            ),
            ask="請他提供逐筆(而非逐月彙總)的原始成交紀錄,以及第三方對帳單。",
        ))

    # ── 2. 一階自相關過高:同樣是平滑化的徵兆 ──
    if f.autocorr_lag1 is not None and n >= 12:
        threshold = 2.0 / (n ** 0.5)
        if f.autocorr_lag1 > threshold:
            findings.append(Finding(
                code="high_autocorr",
                severity="medium",
                what=(
                    f"月報酬的一階自相關為 {f.autocorr_lag1:+.3f},"
                    f"超過隨機下的參考界線 {threshold:.3f}。"
                ),
                means="這個月的報酬能明顯預測下個月 —— 常見於「報酬被跨期分攤」的情形。",
                does_not_mean=(
                    "**不代表造假。** 流動性差的資產(如未上市股權、房地產)"
                    "本來就會因估值延遲而有正自相關。"
                ),
                ask="他持有的資產有公開市價嗎?估值是誰做的、多久做一次?",
            ))

    # ── 3. 夏普值高得罕見:選擇偏差的紅旗(不是造假的證據)──
    if f.sharpe and f.sharpe.n_periods >= 12:
        sr = f.sharpe.sharpe_annualized
        if sr > 3.0:
            findings.append(Finding(
                code="extreme_sharpe",
                severity="high",
                what=(
                    f"年化夏普值約 {sr:.2f}(95% 信賴區間 "
                    f"[{f.sharpe.ci_low:.2f}, {f.sharpe.ci_high:.2f}])。"
                ),
                means=(
                    "這在公開市場策略中極為罕見。而且你會看到它,"
                    "正是因為它被挑出來推銷給你(選擇偏差)—— "
                    "那些夏普值普通的人不會來找你。"
                ),
                does_not_mean=(
                    "**不代表造假,也不代表不可能。** 高頻做市、統計套利的合法策略"
                    "也可能有高夏普。而且若報酬被平滑化,這個夏普值本身就被高估了 —— "
                    "此時真實的不確定性比上面的信賴區間更大。"
                    "(註:此處的「> 3」是實務經驗法則的參考門檻,不是統計檢定的臨界值;"
                    "其意義也隨報酬頻率而變。信賴區間用 Lo (2002) 的 i.i.d. 近似 SE,"
                    "報酬有自相關時會低估不確定性。)"
                ),
                ask="這個夏普是用什麼期間、什麼資產、扣除多少費用後算的?能否提供逐筆紀錄?",
            ))

    # ── 4. 尾數分布:**降為純描述,不再作為可疑旗標**。
    #    理由(對抗驗證結論):尾數是否有意義,取決於資料的量化精度
    #    (四捨五入位數、tick size)—— 我們無從得知,任何門檻都是拍腦袋。
    #    數字仍保留在 terminal_digits 欄位供進階使用者自行判讀,
    #    但不計入可疑程度(與 round_number_ratio 同一待遇)。

    # ── 4b. 零波動 / 極低波動:最露骨的偽造樣態,卻是其他檢定的盲區 ──
    #    報酬完全恆定時,runs test(全同號)與夏普(變異為 0)都回 None,
    #    等於「最假的資料反而一項檢定都跑不了」。必須明確補上這一項。
    if n >= 12 and (vol == 0.0 or (mean != 0 and vol / abs(mean) < 0.02)):
        findings.append(Finding(
            code="zero_volatility",
            severity="high",
            what=(
                f"{n} 期報酬幾乎完全一樣(波動度 {vol:.4%},平均 {mean:.2%})。"
            ),
            means=(
                "真實市場的報酬不可能期期近乎恆定。「每月固定 X%」是龐氏騙局"
                "最典型的樣態 —— 因為那個數字不是市場給的,是人填的。"
            ),
            does_not_mean=(
                "**不代表必然造假。** 定存、貨幣基金的報酬也近乎恆定 —— "
                "但那樣的年化報酬只有 1~5%。恆定又高報酬,才是矛盾所在。"
            ),
            ask="這個報酬是投資績效還是『承諾配息』?錢實際投到哪裡?能否第三方查證?",
        ))

    # ── 5. 月月正報酬:描述性事實(最有力的追問切入點)──
    if n >= 12 and pos_ratio >= 0.95:
        findings.append(Finding(
            code="almost_never_loses",
            severity="high",
            what=f"{n} 期裡有 {pos_ratio:.0%} 是正報酬,幾乎從不虧損。",
            means=(
                "任何承擔市場風險的策略都會有虧損的月份。"
                "「幾乎從不虧損」通常意味著:風險被藏在別的地方"
                "(如厚尾、流動性、對手方風險),或報酬被平滑化。"
            ),
            does_not_mean=(
                "**不代表造假。** 極低風險的套利、或持有短期公債,"
                "也可以長期幾乎不虧。但那樣的報酬率不會很高。"
            ),
            ask=(
                f"平均月報酬 {mean:.2%} 卻幾乎不虧損 —— 請他說明:"
                "風險到底在哪裡?最壞的情況會賠多少?"
            ),
        ))

    # ── 序數化的可疑程度(不給假百分比)──
    high = sum(1 for x in findings if x.severity == "high")
    if n < 8:
        level = "資訊不足"
    elif high >= 2:
        level = "高度可疑"
    elif high == 1:
        level = "可疑"
    elif findings:
        level = "略有疑點"
    else:
        level = "未發現明顯疑點"

    f.findings = findings
    f.suspicion_level = level
    return f


def render_forensics(f: ReturnsForensics, *, label: str = "宣稱的績效") -> str:
    """輸出可讀報告。"""
    L = ["=" * 66, f"        假績效統計鑑識 — {label}", "=" * 66, ""]
    if f.n_periods < 8:
        L.append("⚠️ 資料期數太少(< 8 期),無法做有意義的鑑識。")
        if f.n_dropped:
            L.append(
                f"   (其中 {f.n_dropped} 期為非數值 NaN/inf,已剔除 —— "
                "期數不足可能正是因為這些缺漏)"
            )
        L.append("   請向對方索取更長、更完整的紀錄 —— 拿不出來,本身就是警訊。")
        return "\n".join(L)

    L.append(f"【可疑程度】{f.suspicion_level}")
    L.append("")
    L.append("【基本描述】")
    L.append(f"  期數        : {f.n_periods}")
    if f.n_dropped:
        L.append(
            f"  ⚠ 有 {f.n_dropped} 期為非數值(NaN/inf),已剔除 —— "
            "請追問原始資料為何缺漏"
        )
    L.append(f"  平均報酬    : {f.mean_return:.2%}")
    L.append(f"  波動度      : {f.volatility:.2%}")
    L.append(f"  正報酬比例  : {f.positive_ratio:.0%}")
    if f.sharpe:
        L.append(
            f"  年化夏普    : {f.sharpe.sharpe_annualized:.2f} "
            f"(95% CI [{f.sharpe.ci_low:.2f}, {f.sharpe.ci_high:.2f}])"
        )
    if f.round_number_ratio is not None:
        L.append(f"  尾數為 0 比例: {f.round_number_ratio:.0%}(純描述,無門檻)")
    L.append("")

    if not f.findings:
        L.append("【檢驗結果】未發現明顯的統計疑點。")
        L.append("  但這**不代表這份績效是真的** —— 一份精心捏造的資料")
        L.append("  可以通過所有這些檢驗。唯一可靠的驗證是:第三方對帳單 + 完整逐筆紀錄。")
    else:
        L.append("【發現的可疑徵兆】")
        for x in f.findings:
            tag = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(x.severity, "•")
            L.append(f"  {tag} {x.what}")
            L.append(f"      這代表  :{x.means}")
            L.append(f"      不代表  :{x.does_not_mean}")
            L.append(f"      該追問  :{x.ask}")
            L.append("")

    L.append("─" * 66)
    L.append("重要:本工具**不指控任何人造假**。它只指出「值得追問的統計特徵」。")
    L.append("      唯一能證明績效為真的,是可查證的第三方紀錄。")
    L.append("")
    L.append("註:我們**不使用班佛定律**。報酬有正有負、不跨數量級,")
    L.append("    班佛定律的前提根本不成立,用它只會製造假指控。")
    return "\n".join(L)
