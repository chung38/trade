"""倖存者偏差模擬器 — 用數學拆穿「名師神準」的幻覺。

核心洞見:

    你在群組裡看到的「連贏 10 次的老師」,不需要任何預測能力就會出現。
    只要人夠多,**純靠擲硬幣**也很可能有人連贏 10 次 —— 而那個人會被
    推出來當「名師」,其他 999 個賠錢退場的人你永遠看不到。

    這就是倖存者偏差:你看到的不是「有能力的人」,是「被篩選出來的幸運兒」。

本模組提供**精確的解析解**(不只是模擬),讓這個論證無可辯駁。

⚠️ 語氣紀律:一律用機率陳述(「有 62% 的機率至少一人…」),
   絕不說「必然存在」。實測 K=1000、N=10、p=0.5 時,P(至少一人) = 0.6236,
   並不是 1。用確定性語氣就是我們自己在說假話。
"""

from __future__ import annotations

from dataclasses import dataclass


def prob_streak_in_trials(n_trials: int, streak: int, p_win: float = 0.5) -> float:
    """單一交易者在 n_trials 次交易中,出現「至少一次連續 streak 勝」的機率。

    用動態規劃精確計算(不是近似、不是模擬):

        設 f[i][j] = 前 i 次交易、目前結尾連勝長度為 j、且「尚未」達成過
        目標連勝的機率。一旦連勝達到 streak 就吸收到「已達成」狀態。

        轉移:
            贏(機率 p):j → j+1;若 j+1 == streak 則進入吸收態
            輸(機率 1-p):j → 0

        答案 = 吸收態的總機率。

    這是標準的 run 問題(Feller,《機率論及其應用》)。
    """
    if streak <= 0:
        return 1.0
    if n_trials < streak:
        return 0.0
    if not (0.0 <= p_win <= 1.0):
        raise ValueError("p_win 必須在 [0, 1]")
    if p_win == 0.0:
        return 0.0
    if p_win == 1.0:
        return 1.0

    q = 1.0 - p_win
    # states[j] = 尚未達成、目前結尾連勝 j 的機率(j = 0..streak-1)
    states = [0.0] * streak
    states[0] = 1.0
    achieved = 0.0

    for _ in range(n_trials):
        new = [0.0] * streak
        # 輸:所有未吸收狀態都回到 j=0
        new[0] = q * sum(states)
        # 贏:j → j+1;j = streak-1 時吸收
        for j in range(streak - 1):
            new[j + 1] += p_win * states[j]
        achieved += p_win * states[streak - 1]
        states = new

    return achieved


def prob_at_least_one_streak(
    n_traders: int, n_trials: int, streak: int, p_win: float = 0.5
) -> float:
    """K 個獨立的交易者中,「至少一人」出現連續 streak 勝的機率。

        P(至少一人) = 1 - (1 - p_single)^K

    假設各交易者獨立。這在「大家交易不同標的」時合理;
    若都跟同一支股票,實際相關性會讓這個數字偏高(我們的估計偏保守)。
    """
    if n_traders <= 0:
        return 0.0
    p_single = prob_streak_in_trials(n_trials, streak, p_win)
    if p_single <= 0.0:
        return 0.0
    if p_single >= 1.0:
        return 1.0
    return 1.0 - (1.0 - p_single) ** n_traders


def expected_number_with_streak(
    n_traders: int, n_trials: int, streak: int, p_win: float = 0.5
) -> float:
    """期望有多少人會達成該連勝(= K × 單人機率,期望值的線性性)。"""
    return n_traders * prob_streak_in_trials(n_trials, streak, p_win)


def prob_perfect_run(streak: int, p_win: float = 0.5) -> float:
    """「連續 streak 次全對」的機率(N-for-N 模型):p^streak。

    這對應「老師開了 streak 支,支支都漲」的宣稱。
    """
    if streak <= 0:
        return 1.0
    return p_win ** streak


@dataclass(frozen=True)
class GuruIllusion:
    """『名師神準』的倖存者偏差分析結果。"""

    n_traders: int
    n_trials: int
    streak: int
    p_win: float

    prob_single: float          # 單人達成該連勝的機率
    prob_at_least_one: float    # 至少一人達成的機率
    expected_count: float       # 期望達成人數

    headline: str
    explanation: list[str]

    def as_dict(self) -> dict:
        return {
            "n_traders": self.n_traders,
            "n_trials": self.n_trials,
            "streak": self.streak,
            "p_win": self.p_win,
            "prob_single": self.prob_single,
            "prob_at_least_one": self.prob_at_least_one,
            "expected_count": self.expected_count,
            "headline": self.headline,
            "explanation": self.explanation,
        }


def guru_illusion(
    *,
    n_traders: int = 1000,
    n_trials: int = 20,
    streak: int = 10,
    p_win: float = 0.5,
) -> GuruIllusion:
    """旗艦反詐場景:「一個 1000 人的群組裡,出現一個連贏 10 次的老師」有多正常?

    Args:
        n_traders: 群組人數(或「市面上有多少個自稱老師的人」)
        n_trials:  每人做了幾次預測
        streak:    宣稱的連勝次數
        p_win:     單次預測正確的機率。預設 0.5(純猜漲跌)。
                   ⚠️ 這是**虛無假設**:假設老師毫無能力。我們算的是
                   「在他完全沒能力的前提下,這種績效有多容易出現」。
    """
    p_single = prob_streak_in_trials(n_trials, streak, p_win)
    p_any = prob_at_least_one_streak(n_traders, n_trials, streak, p_win)
    expected = expected_number_with_streak(n_traders, n_trials, streak, p_win)

    headline = (
        f"就算 {n_traders} 個人全都只是在擲硬幣(毫無預測能力),"
        f"其中出現至少一個『連贏 {streak} 次』的人,機率是 {p_any:.1%}。"
    )

    explanation = [
        f"單獨一個人在 {n_trials} 次裡連贏 {streak} 次的機率只有 {p_single:.4%} —— 看起來很神。",
        f"但 {n_traders} 個人一起試,期望會有 {expected:.1f} 個人達成。",
        "那個幸運兒會被推出來當『名師』,貼出他的連勝截圖;"
        "其他人默默賠錢退場,你永遠看不到他們。",
        "這就是倖存者偏差 —— 你看到的不是能力,是被篩選出來的運氣。",
        f"要判斷這位『老師』是真有本事,不能看他挑給你的 {streak} 次,"
        "而要看他**完整、連續、夠長**的全部紀錄(含失敗的)。",
    ]

    if p_any >= 0.5:
        # 用 .1% 而非 .0%:0.997 印成「100%」會讀起來像確定性,那是說假話。
        explanation.insert(
            0,
            f"⚠️ 這種『神績效』在純運氣下出現的機率超過一半({p_any:.1%})—— "
            "它根本不是證據。",
        )

    return GuruIllusion(
        n_traders=n_traders,
        n_trials=n_trials,
        streak=streak,
        p_win=p_win,
        prob_single=p_single,
        prob_at_least_one=p_any,
        expected_count=expected,
        headline=headline,
        explanation=explanation,
    )


def render_guru_illusion(g: GuruIllusion) -> str:
    """輸出可直接放進報告 / scam-check 的文字。

    標題刻意**避免必然式措辭** —— 實測 K=1000、N=10 時機率是 99.7%,
    不是 1。用確定性語氣就是我們自己在說假話。
    """
    lines = [
        "【🎲 倖存者偏差:「連贏 N 次的神人」有多容易靠運氣出現?】",
        f"  {g.headline}",
        "",
    ]
    for e in g.explanation:
        lines.append(f"  • {e}")
    lines.append("")
    lines.append(
        "  註:以上假設每個人都毫無預測能力(勝率 50%)。"
        "算的是「在他沒能力的前提下,這種績效有多容易發生」。"
    )
    return "\n".join(lines)
