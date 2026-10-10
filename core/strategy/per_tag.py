"""逐策略(per-tag)完整裁決、反事實分析、跟單抽算。

這是把工具從「整體體檢」升級成「精準定位」的關鍵:
  - per_tag_verdicts:對每個策略標籤產出「描述統計」(期望值/勝率/獲利因子),
    刻意**不做**逐標籤顯著性裁決 —— 同時檢定 K 個標籤會把運氣認成優勢
    (多重比較問題),所以只告訴使用者『哪一招在送錢』,不認證『哪一招有優勢』。
  - counterfactual:如果停掉表現最差的那一招,整體會變怎樣。
  - follow_the_guru:把『聽老師/明牌/跟單』的交易單獨抽出來算期望值 ——
    兌現反詐金句裡『把老師推薦記下來算期望值,幾乎都是負的』的承諾。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..antiscam.signals import _FOLLOW_KEYWORDS
from ..metrics.performance import compute_metrics
from ..models import TradeLog
from ..verdict.judge import VerdictLevel, judge


@dataclass
class TagVerdict:
    """單一策略標籤的『描述統計』摘要。

    刻意**不做顯著性檢定、不頒發優勢徽章**。原因:

      對 K 個標籤各跑一次顯著性檢定卻不做多重比較校正,會讓「至少一個標籤
      被誤判為具優勢」的機率隨 K 膨脹(蒙地卡羅實測:K=5 → 24%,K=10 → 41%,重現 experiments/exp_multiple_comparison.py;
      遠超名目的 5%)。對一個宗旨是「戳破運氣幻覺」的工具,自己對純隨機的
      雜訊頒發「🟩 具優勢」徽章,是直接的自我矛盾。

      而 per-tag 樣本通常只有 5~30 筆,做 Holm/Bonferroni 校正後幾乎永遠
      不顯著 —— 與其給一個沒有資訊量的「不顯著」,不如誠實地只報描述統計,
      讓使用者看「哪一招在賠錢」,而不是「哪一招被認證為有優勢」。

      整體裁決(整份紀錄只做一次檢定)仍保留完整的顯著性判定。
    """

    tag: str
    n_trades: int
    expectancy: float
    win_rate: float
    profit_factor: float
    total_pnl: float
    low_sample: bool          # 樣本太少,結論僅供參考

    @property
    def is_losing(self) -> bool:
        return self.expectancy < 0

    @property
    def descriptor(self) -> str:
        """白話描述(非優勢認證)。"""
        base = "🔻 賠錢中" if self.is_losing else "🔺 賺錢中"
        return f"{base}(樣本少)" if self.low_sample else base


@dataclass
class CounterfactualResult:
    """反事實分析:停掉最差策略後的『會計式』前後對照。

    刻意只做加減法(總損益、每筆期望值的前後差),**不重新下裁決、
    不宣稱「不做這一招你就有救了」**。原因:最差的標籤是「從同一份資料裡挑出來的」,
    挑完再對同一份資料重新檢定,屬於資料探勘(data dredging);其改善有很大一部分
    來自回歸均值,把它講成「有救」是過度樂觀的認證。
    """

    worst_tag: str
    before_expectancy: float
    after_expectancy: float
    before_total_pnl: float
    after_total_pnl: float
    message: str = ""


@dataclass
class FollowGuruResult:
    """跟單 / 聽明牌交易的專屬裁決。"""

    n_trades: int
    expectancy: float
    total_pnl: float
    win_rate: float
    level: VerdictLevel
    follow_tags: list[str] = field(default_factory=list)
    message: str = ""


def per_tag_verdicts(
    log: TradeLog,
    *,
    min_tag_trades: int = 5,
    n_bootstrap: int = 1500,   # 保留參數以維持相容;不再用於顯著性檢定
) -> list[TagVerdict]:
    """對每個策略標籤產出『描述統計』,依期望值由低到高排序。

    最差的排最前面 —— 因為使用者最需要先看到『該砍哪一招』。

    刻意不做顯著性檢定(見 TagVerdict 的說明):對多個標籤各檢定一次卻不做
    多重比較校正,會系統性地把運氣誤認為優勢。這裡只誠實呈現「這一招賺或賠、
    賠多少、樣本夠不夠」,把「是不是真優勢」的判斷留給整體裁決。
    """
    tags = {t.tag for t in log if t.tag}
    results: list[TagVerdict] = []
    for tag in tags:
        sub = log.filter_by_tag(tag)
        if len(sub) < 2:
            continue
        m = compute_metrics(sub)
        results.append(TagVerdict(
            tag=tag,
            n_trades=len(sub),
            expectancy=m.expectancy,
            win_rate=m.win_rate,
            profit_factor=m.profit_factor,
            total_pnl=m.total_pnl,
            low_sample=len(sub) < 30,
        ))
    results.sort(key=lambda r: r.expectancy)   # 最差(最該砍)的排最前
    return results


def counterfactual_drop_worst(
    log: TradeLog,
    *,
    tag_verdicts: list[TagVerdict] | None = None,
    n_bootstrap: int = 2000,
) -> CounterfactualResult | None:
    """如果停掉『期望值最差』的策略標籤,整體會變怎樣。

    回傳 None 表示無法分析(無 tag、或只有一個 tag、或砍了就沒交易了)。
    """
    tv = tag_verdicts if tag_verdicts is not None else per_tag_verdicts(log)
    if len(tv) < 2:
        return None   # 至少要有兩個 tag 才談「砍掉一個」

    worst = tv[0]
    if worst.expectancy >= 0:
        return None   # 最差的都沒在送錢,沒必要建議砍

    before_m = compute_metrics(log)

    after_log = log.filter(lambda t: t.tag != worst.tag, label=f"drop_{worst.tag}")
    if len(after_log) < 2:
        return None
    after_m = compute_metrics(after_log)

    # 純會計式對照:只陳述「回測期間的加減法」,不重新下裁決、不宣稱「有救」。
    delta_exp = after_m.expectancy - before_m.expectancy
    delta_pnl = after_m.total_pnl - before_m.total_pnl
    msg = (
        f"在這段回測期間,若當初沒做『{worst.tag}』這一招:"
        f"總損益會從 {before_m.total_pnl:+,.2f} 變成 {after_m.total_pnl:+,.2f}"
        f"(差 {delta_pnl:+,.2f});每筆期望值從 {before_m.expectancy:+.2f} "
        f"變成 {after_m.expectancy:+.2f}(改善 {delta_exp:+.2f})。"
        " 注意:這是『事後從同一份資料挑出最差的一招』再回頭算的假設情境,"
        "改善有一部分來自回歸均值,不代表未來停掉它就一定會賺。"
    )

    return CounterfactualResult(
        worst_tag=worst.tag,
        before_expectancy=before_m.expectancy,
        after_expectancy=after_m.expectancy,
        before_total_pnl=before_m.total_pnl,
        after_total_pnl=after_m.total_pnl,
        message=msg,
    )


def follow_the_guru(
    log: TradeLog,
    *,
    n_bootstrap: int = 2000,
) -> FollowGuruResult | None:
    """把『聽老師 / 明牌 / 跟單 / VIP 群』的交易單獨抽出來算期望值。

    這兌現了反詐金句裡的承諾:『把老師的歷史推薦全部記下來、算期望值 ——
    多數是負的。』用使用者自己的數字檢驗跟單績效 —— 不定罪,讓數字說話。

    回傳 None 表示沒有可辨識的跟單交易。
    """
    def is_follow(t) -> bool:
        return bool(t.tag) and any(k in str(t.tag).lower() for k in _FOLLOW_KEYWORDS)

    sub = log.filter(is_follow, label="follow_guru")
    if len(sub) < 2:
        return None

    follow_tags = sorted({t.tag for t in sub if t.tag})
    m = compute_metrics(sub)
    # 用與整體裁決相同的樣本量門檻(不放寬到 5):跟單交易若只有幾筆,
    # 不該被認證為「具優勢」,而應誠實回報樣本不足。
    v = judge(sub, metrics=m, n_bootstrap=n_bootstrap)

    if m.expectancy < 0:
        msg = (
            f"你『聽老師 / 跟單 / 明牌』的 {len(sub)} 筆交易,"
            f"每筆平均賠 {abs(m.expectancy):.2f},合計 {m.total_pnl:+.2f}。"
            "這是用你自己的錢算出來的數字 —— 這段紀錄裡,跟單不但沒讓你賺,還在穩定地讓你賠。"
            "那些『老師』真正賺的,是你的學費與群費,不是市場。"
        )
    else:
        msg = (
            f"你『聽老師 / 跟單 / 明牌』的 {len(sub)} 筆交易,期望值為 "
            f"{m.expectancy:+.2f}。即使帳面為正,也要警覺:這可能只是運氣,"
            "而且你看到的『老師神準』往往是倖存者偏差。請持續用統計檢驗,別輕信。"
        )

    return FollowGuruResult(
        n_trades=len(sub),
        expectancy=m.expectancy,
        total_pnl=m.total_pnl,
        win_rate=m.win_rate,
        level=v.level,
        follow_tags=follow_tags,
        message=msg,
    )
