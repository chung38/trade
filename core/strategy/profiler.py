"""交易模式剖析:從歷史交易反推「使用者實際上在做什麼」。

我們無法讀心,但可以從交易的客觀特徵(持倉時間、方向偏好、
標的集中度、進出場時間分布、tag 標籤……)推斷出交易風格,
作為產生策略骨架的依據。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..metrics.performance import INTRADAY_RATIO_THRESHOLD
from ..models import Market, Side, TradeLog


@dataclass
class StrategyProfile:
    """一份交易紀錄反推出的策略輪廓。"""

    style: str                              # 推斷的交易風格(中文描述)
    style_code: str                         # 機器可讀風格碼
    dominant_market: Market                 # 主要市場
    dominant_side: Side                     # 主要方向
    avg_holding_days: float
    median_holding_days: float
    symbol_concentration: float             # 前三大標的佔交易比例
    distinct_symbols: int
    tags: dict[str, int] = field(default_factory=dict)   # tag → 次數
    per_tag_winrate: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    timing_known_trades: int = 0
    timing_unknown_trades: int = 0
    timing_data_complete: bool = True
    timing_metrics_available: bool = False
    side_known_trades: int = 0
    side_unknown_trades: int = 0
    side_data_complete: bool = True
    side_metrics_available: bool = False

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["dominant_market"] = self.dominant_market.value
        d["dominant_side"] = self.dominant_side.value
        return d


def _classify_style(avg_days: float, intraday_ratio: float) -> tuple[str, str]:
    """依平均持倉天數與當沖比例分類交易風格。"""
    if intraday_ratio >= INTRADAY_RATIO_THRESHOLD:
        return ("當沖 / 極短線(高頻、高成本、最接近賭博的型態)", "scalp_intraday")
    if avg_days < 5:
        return ("短線波段(數日內進出)", "swing_short")
    if avg_days < 30:
        return ("中期波段(數週)", "swing_medium")
    if avg_days < 180:
        return ("中長期持有(數月)", "position")
    return ("長期投資(半年以上)", "long_term")


def profile_strategy(log: TradeLog) -> StrategyProfile:
    """剖析交易紀錄,產出策略輪廓。"""
    trades = list(log)
    n = len(trades)

    if n == 0:
        return StrategyProfile(
            style="無資料", style_code="empty",
            dominant_market=Market.UNKNOWN, dominant_side=Side.LONG,
            avg_holding_days=0, median_holding_days=0,
            symbol_concentration=0, distinct_symbols=0,
        )

    import statistics

    timing_known = [
        t for t in trades
        if t.entry_time_known
        and t.exit_time_known
        and getattr(t, "time_basis_consistent", True)
    ]
    timing_known_n = len(timing_known)
    timing_unknown_n = n - timing_known_n
    holding = sorted(
        days for t in timing_known
        if (days := t.holding_days) is not None
    )
    avg_days = sum(holding) / len(holding) if holding else 0.0
    # 偶數筆要取中央兩值的平均:持倉 1 天與 9 天的中位數是 5 天,不是 9 天
    median_days = float(statistics.median(holding)) if holding else 0.0
    # 當沖比例的分子/分母都只使用雙時間已知的交易。
    intraday_ratio = (
        sum(1 for t in timing_known if t.is_day_trade) / timing_known_n
        if timing_known_n else 0.0
    )

    market_counts = Counter(t.market for t in trades)
    side_counts = Counter(t.side for t in trades if t.side_known)
    symbol_counts = Counter(t.symbol for t in trades)

    dominant_market = market_counts.most_common(1)[0][0]
    # dominant_side 保留 Side 型別以維持 API 相容；side_metrics_available=False
    # 時只是佔位值，報告/骨架不得把它描述成使用者的方向偏好。
    dominant_side = side_counts.most_common(1)[0][0] if side_counts else Side.LONG
    distinct_symbols = len(symbol_counts)
    top3 = sum(c for _, c in symbol_counts.most_common(3))
    concentration = top3 / n

    if timing_known_n and timing_unknown_n == 0:
        style, style_code = _classify_style(avg_days, intraday_ratio)
    elif timing_known_n:
        style = "交易風格無法判斷(進出場時間不完整)"
        style_code = "timing_incomplete"
    else:
        style = "交易風格無法判斷(缺少進出場時間)"
        style_code = "timing_unavailable"

    # tag 統計 + 各 tag 的勝率(這是反推「哪套邏輯有效」的關鍵)
    tag_counts: Counter[str] = Counter()
    tag_wins: Counter[str] = Counter()
    for t in trades:
        if t.tag:
            tag_counts[t.tag] += 1
            if t.is_win:
                tag_wins[t.tag] += 1
    per_tag_winrate = {
        tag: tag_wins[tag] / cnt for tag, cnt in tag_counts.items() if cnt > 0
    }

    notes: list[str] = []
    if timing_unknown_n:
        if timing_known_n:
            notes.append(
                f"僅 {timing_known_n}/{n} 筆同時有進出場時間;"
                "持倉天數與風格只根據已知子集,可能有選擇性缺值偏差。"
            )
        else:
            notes.append(
                "全部交易都缺少可靠的進/出場時間;"
                "不得把佔位日期當成當沖,本次不判定持倉風格。"
            )
    side_known_n = sum(side_counts.values())
    side_unknown_n = n - side_known_n
    if side_unknown_n:
        notes.append(
            f"{side_unknown_n}/{n} 筆沒有可靠的做多/做空方向；"
            "方向偏好只使用已明示方向的交易，若全部缺漏則不判定。"
        )
    if concentration > 0.7 and distinct_symbols < 5:
        notes.append(
            f"交易高度集中在少數標的(前三大佔 {concentration:.0%})。"
            "績效可能綁定特定標的的單一行情,換標的未必複製得了。"
        )
    if not tag_counts:
        notes.append(
            "資料中沒有策略標籤(tag)。建議未來每筆交易標註進場理由,"
            "才能分辨是哪一套邏輯真正帶來獲利。"
        )
    if timing_unknown_n == 0 and intraday_ratio >= INTRADAY_RATIO_THRESHOLD:
        notes.append(
            "以當沖為主:這類交易長期能穩定獲利的比例極低,且成本侵蝕嚴重。"
        )

    return StrategyProfile(
        style=style,
        style_code=style_code,
        dominant_market=dominant_market,
        dominant_side=dominant_side,
        avg_holding_days=avg_days,
        median_holding_days=median_days,
        symbol_concentration=concentration,
        distinct_symbols=distinct_symbols,
        tags=dict(tag_counts),
        per_tag_winrate=per_tag_winrate,
        notes=notes,
        timing_known_trades=timing_known_n,
        timing_unknown_trades=timing_unknown_n,
        timing_data_complete=timing_unknown_n == 0,
        timing_metrics_available=bool(timing_known_n),
        side_known_trades=side_known_n,
        side_unknown_trades=side_unknown_n,
        side_data_complete=side_unknown_n == 0,
        side_metrics_available=bool(side_known_n),
    )
