"""鑑識用的統計原語 — 純標準庫,不依賴 scipy。

提供:
    chi2_sf(x, k)  卡方分布的存活函數 P(X > x)
    normal_sf(z)   標準常態的存活函數 P(Z > z)
    runs_test      連段檢定(偵測「過度平滑」的報酬序列)
    lag1_autocorr  一階自相關
    sharpe_with_ci 夏普值與 Lo (2002) 校正的標準誤 / 信賴區間
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def normal_sf(z: float) -> float:
    """標準常態的上尾機率 P(Z > z)。"""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _gammap_series(a: float, x: float, *, max_iter: int = 500,
                   eps: float = 1e-14) -> float:
    """下不完全 gamma 的正則化 P(a, x),級數展開(適用 x < a+1)。"""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(max_iter):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaq_cf(a: float, x: float, *, max_iter: int = 500,
               eps: float = 1e-14) -> float:
    """上不完全 gamma 的正則化 Q(a, x),連分數展開(適用 x >= a+1)。"""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_q(a: float, x: float) -> float:
    """正則化上不完全 gamma 函數 Q(a, x) = 1 - P(a, x)。"""
    if x < 0 or a <= 0:
        raise ValueError("gamma_q 需要 a > 0 且 x >= 0")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gammap_series(a, x)
    return _gammaq_cf(a, x)


def chi2_sf(x: float, df: int) -> float:
    """卡方分布的存活函數 P(X > x),df 個自由度。

        P(X > x) = Q(df/2, x/2)
    """
    if df <= 0:
        raise ValueError("自由度必須 > 0")
    if x <= 0:
        return 1.0
    return gamma_q(df / 2.0, x / 2.0)


# ── 連段檢定(Wald–Wolfowitz runs test)──────────────────────
@dataclass(frozen=True)
class RunsTestResult:
    """連段檢定結果。用於偵測報酬序列是否「過度平滑」。"""

    n_positive: int
    n_negative: int
    runs: int
    expected_runs: float
    z: float
    p_two_sided: float
    too_few_runs: bool     # 連段太少 = 同號聚集 = 平滑/趨勢性
    note: str = ""


def runs_test(values: list[float]) -> RunsTestResult | None:
    """檢定「正負號的排列是否像隨機」。

    連段(run)= 連續同號的一段。若報酬被人為平滑(如龐氏騙局按月給固定報酬),
    正號會聚集,連段數會**顯著偏少**。

    虛無假設:正負號的排列是隨機的(給定正負個數)。
        E[R]   = 2·n1·n2/n + 1
        Var[R] = 2·n1·n2·(2·n1·n2 − n) / (n²·(n−1))

    ⚠️ 這個檢定只說「排列不像隨機」,**不說「造假」**。
       趨勢性市場、動量策略都會自然產生連段偏少。
    回傳 None 表示樣本不足或全部同號(檢定無定義)。
    """
    vals = [v for v in values if v != 0]
    n = len(vals)
    if n < 8:
        return None
    signs = [v > 0 for v in vals]
    n1 = sum(signs)
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        return None   # 全正或全負,連段檢定無定義

    runs = 1 + sum(1 for i in range(1, n) if signs[i] != signs[i - 1])
    exp_r = 2.0 * n1 * n2 / n + 1.0
    var_r = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    if var_r <= 0:
        return None
    z = (runs - exp_r) / math.sqrt(var_r)
    p = 2.0 * normal_sf(abs(z))

    return RunsTestResult(
        n_positive=n1,
        n_negative=n2,
        runs=runs,
        expected_runs=exp_r,
        z=z,
        p_two_sided=min(1.0, p),
        too_few_runs=(z < 0),
    )


def lag1_autocorr(values: list[float]) -> float | None:
    """一階自相關係數。被平滑化的報酬序列常有明顯的正自相關。

    在虛無假設(獨立)下,標準誤約 1/sqrt(n),故 |r1| > 2/sqrt(n) 值得注意。
    回傳 None 表示樣本不足或變異為 0。
    """
    n = len(values)
    if n < 4:
        return None
    m = sum(values) / n
    denom = sum((x - m) ** 2 for x in values)
    if denom == 0:
        return None
    num = sum((values[i] - m) * (values[i + 1] - m) for i in range(n - 1))
    return num / denom


# ── 夏普值與 Lo (2002) 校正的信賴區間 ────────────────────────
@dataclass(frozen=True)
class SharpeResult:
    """夏普值與其不確定性。"""

    sharpe_per_period: float
    sharpe_annualized: float
    periods_per_year: int
    n_periods: int
    std_error: float          # Lo (2002) 的 SE(單期);已依 sqrt(ppy) 年化
    ci_low: float
    ci_high: float
    note: str = ""


def sharpe_with_ci(
    returns: list[float],
    *,
    periods_per_year: int = 12,
    risk_free_per_period: float = 0.0,
) -> SharpeResult | None:
    """計算夏普值,並用 Lo (2002) 的公式給出標準誤與 95% 信賴區間。

        SE(SR) ≈ sqrt((1 + SR²/2) / n)          (i.i.d. 假設下)

    ⚠️ 誠實揭露:
      - 這個 SE 只在「報酬 i.i.d.」時成立。被平滑化的報酬序列有正自相關,
        會讓夏普值被**高估**,而 SE 被低估 —— 也就是說,對可疑資料,
        真實的不確定性比這個區間更大。
      - **夏普值高不等於造假。** 它可疑,是因為你看到的是被推銷的那一個
        (選擇偏差)。合法策略也可能有高夏普。
    回傳 None 表示樣本不足或報酬無變異。
    """
    n = len(returns)
    if n < 4:
        return None
    excess = [r - risk_free_per_period for r in returns]
    m = sum(excess) / n
    var = sum((x - m) ** 2 for x in excess) / (n - 1)
    if var <= 0:
        return None
    sd = math.sqrt(var)
    sr = m / sd
    ppy = max(1, periods_per_year)
    sr_ann = sr * math.sqrt(ppy)

    se = math.sqrt((1.0 + 0.5 * sr * sr) / n)      # Lo (2002),單期
    se_ann = se * math.sqrt(ppy)

    return SharpeResult(
        sharpe_per_period=sr,
        sharpe_annualized=sr_ann,
        periods_per_year=ppy,
        n_periods=n,
        std_error=se_ann,
        ci_low=sr_ann - 1.96 * se_ann,
        ci_high=sr_ann + 1.96 * se_ann,
    )


# ── 尾數(最後一位數字)適合度檢定 ──────────────────────────
@dataclass(frozen=True)
class TerminalDigitResult:
    """尾數卡方適合度檢定。"""

    counts: list[int]        # 0~9 的出現次數
    n: int
    chi2: float
    df: int
    p_value: float
    note: str = ""


def terminal_digit_test(values: list[float], *, decimals: int = 2
                        ) -> TerminalDigitResult | None:
    """檢定「數字的最後一位」是否均勻分布於 0~9。

    捏造的數字常偏好某些尾數(如 0、5)。這個檢定用卡方適合度。

    ⚠️ 誠實揭露:尾數偏離均勻**不等於造假**。最小跳動點(tick size)、
       四捨五入、報價慣例都會造成偏離。這只是「值得追問來源」的徵兆。
    回傳 None 表示樣本不足(卡方需要每格期望次數夠大)。
    """
    if len(values) < 30:
        return None   # 期望每格 3 次以上才夠;30 筆是最低門檻
    counts = [0] * 10
    scale = 10 ** decimals
    for v in values:
        d = int(abs(round(v * scale))) % 10
        counts[d] += 1
    n = sum(counts)
    if n == 0:
        return None
    expected = n / 10.0
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    df = 9
    return TerminalDigitResult(
        counts=counts, n=n, chi2=chi2, df=df, p_value=chi2_sf(chi2, df)
    )
