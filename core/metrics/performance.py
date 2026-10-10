"""核心績效指標。

這裡計算的不只是「賺多少」,更重要的是「這份獲利的品質與穩定度」。
單看勝率會騙人(高勝率可能搭配巨大虧損),單看總獲利也會騙人
(可能來自一兩筆幸運的暴賺)。所以我們同時看多個互補的角度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..markets import infer_pnl_currency, is_leveraged
from ..models import Market, Side, TradeLog

# 當沖/極短線的判定門檻:當沖交易佔比超過此值,視為「以當沖為主」。
# 集中定義,供 performance 與 profiler 共用(避免魔術數字 0.7 散落多處)。
INTRADAY_RATIO_THRESHOLD = 0.7


@dataclass
class PerformanceMetrics:
    """一份交易紀錄的完整績效畫像。"""

    # ── 基本計數 ────────────────────────────────
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0

    # ── 勝率與盈虧比 ─────────────────────────────
    win_rate: float = 0.0              # 勝率 = 獲利筆數 / 總筆數
    avg_win: float = 0.0              # 平均每筆獲利金額
    avg_loss: float = 0.0            # 平均每筆虧損金額(取正值)
    payoff_ratio: float = 0.0        # 盈虧比 = 平均獲利 / 平均虧損
    profit_factor: float = 0.0       # 獲利因子 = 總獲利 / 總虧損

    # ── 期望值(最關鍵的單一數字)────────────────
    expectancy: float = 0.0          # 每筆交易的期望損益(金額)
    expectancy_r: float = 0.0        # 以 R 為單位的期望值(平均 R-multiple)

    # ── 總體報酬 ────────────────────────────────
    total_pnl: float = 0.0
    total_fees: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    # ── 風險指標 ────────────────────────────────
    max_drawdown: float = 0.0        # 最大回撤(金額)
    max_drawdown_pct: float = 0.0    # 最大回撤(相對峰值百分比)
    max_consecutive_losses: int = 0  # 最長連續虧損次數
    sharpe: float = 0.0              # 夏普值(以每筆交易報酬計,非年化)
    sortino: float = 0.0            # 索提諾值(只懲罰下行波動)

    # ── R-multiple 分布(用於判斷是否「靠少數暴賺撐場」)──
    r_multiples: list[float] = field(default_factory=list)
    largest_win: float = 0.0
    largest_loss: float = 0.0
    top_trade_pnl_share: float = 0.0  # 最賺那一筆佔總獲利的比例

    # ── 交易風格 ────────────────────────────────
    avg_holding_days: float = 0.0
    intraday_ratio: float = 0.0
    is_mostly_intraday: bool = False  # 是否多為當沖/極短線(影響賭博判斷)
    timing_known_trades: int = 0
    timing_unknown_trades: int = 0
    timing_data_complete: bool = True
    timing_metrics_available: bool = False
    timing_note: str = ""

    # 回撤、連虧與累積曲線需要可靠的出場順序；報酬率/夏普需要每筆
    # 名目部位都可信。任一缺失時不得用 0 或佔位值假裝有算。
    sequence_metrics_reliable: bool = True
    sequence_note: str = ""
    return_metrics_reliable: bool = True
    return_note: str = ""
    drawdown_note: str = ""
    pnl_currency: str = ""
    currency_reliable: bool = True
    currency_note: str = ""

    # 回撤百分比是否可靠:pnl-only 資料無資本基準時為 False,
    # 顯示層必須說「無法計算」而不是印假的 0%
    drawdown_pct_reliable: bool = True

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        # r_multiples 通常很長,摘要即可
        d["r_multiples_count"] = len(self.r_multiples)
        d.pop("r_multiples", None)
        # JSON 沒有 Infinity(json.dumps 會輸出非標準的 Infinity 字面值,
        # 下游解析器會炸)—— inf 語意是「不適用」,序列化成 null
        for k, v in list(d.items()):
            if isinstance(v, float) and math.isinf(v):
                d[k] = None
        return d


def _safe_div(a: float, b: float) -> float:
    """「缺資料歸零」型除法:比例/平均值在分母 0 時回 0(如勝率、平均)。"""
    return a / b if b else 0.0


def _ratio(num: float, den: float) -> float:
    """「上界無限」型除法:盈虧比/獲利因子/夏普這類比率,分母 0 且分子為正
    的語意是「無虧損(無下行),不適用/無上界」—— 收成 0 會把最好的樣本
    顯示成最差(0 = 沒有獲利因子),是數字說假話。回傳 math.inf,
    由顯示層轉成「∞(無虧損)」、JSON 層轉 null。
    """
    if den:
        return num / den
    if num > 0:
        return math.inf
    if num < 0:
        return -math.inf
    return 0.0


def fmt_ratio(x: float, decimals: int = 2) -> str:
    """比率顯示:inf → 「∞(無虧損/無下行)」,絕不印成 0 或 inf 字樣。"""
    if math.isinf(x):
        return "∞(無虧損/無下行)" if x > 0 else "N/A"
    return f"{x:.{decimals}f}"


def compute_metrics(log: TradeLog) -> PerformanceMetrics:
    """從交易紀錄計算完整績效指標。"""
    m = PerformanceMetrics()
    raw_trades = list(log)
    all_exit_times_known = all(
        getattr(t, "exit_time_known", True) for t in raw_trades
    )
    exit_times = [t.exit_time for t in raw_trades] if all_exit_times_known else []
    exit_order_is_unique = len(exit_times) == len(set(exit_times))
    time_awareness = {
        bool(t.exit_time.tzinfo is not None and t.exit_time.utcoffset() is not None)
        for t in raw_trades if getattr(t, "exit_time_known", True)
    }
    time_basis_consistent = len(time_awareness) <= 1
    m.sequence_metrics_reliable = (
        all_exit_times_known and exit_order_is_unique and time_basis_consistent
    )
    if not all_exit_times_known:
        m.sequence_note = (
            "部分交易缺少可靠的出場時間；最大回撤、最長連虧與累積曲線無法計算。"
        )
    elif not time_basis_consistent:
        m.sequence_note = (
            "出場時間混用有時區與無時區格式，無法安全排序；"
            "最大回撤、最長連虧與逐筆累積曲線無法計算。"
        )
    elif not exit_order_is_unique:
        m.sequence_note = (
            "多筆交易具有相同出場時間，群組內真實先後未知；"
            "最大回撤、最長連虧與累積曲線無法計算。"
        )
    trades = (
        list(log.sorted_by_time()) if m.sequence_metrics_reliable else raw_trades
    )
    m.total_trades = len(trades)
    if not trades:
        return m

    if any(
        t.side == Side.SHORT
        and t.market in (Market.TW_STOCK, Market.TW_ETF, Market.US_STOCK)
        and not getattr(t, "pnl_is_direct", False)
        for t in trades
    ):
        raise ValueError(
            "放空交易若只用價差推算，會漏掉借券/融券費、利息與召回成本；"
            "請提供券商已扣全部成本的 direct net pnl 後再分析。"
        )

    pnls = [t.pnl or 0.0 for t in trades]
    m.return_metrics_reliable = all(
        getattr(t, "notional_reliable", True)
        and getattr(t, "contract_multiplier_known", True)
        and t.contract_value > 0
        # direct pnl 是帳戶結算後金額；沒明示幣別時，不能假定它與
        # 商品報價/名目本金同幣，否則會製造看似精確的假報酬率。
        and not (
            getattr(t, "pnl_is_direct", False)
            and not getattr(t, "pnl_currency", None)
        )
        and not (
            getattr(t, "pnl_currency", None)
            and infer_pnl_currency(t.symbol, t.market)
            and str(t.pnl_currency).upper() != infer_pnl_currency(t.symbol, t.market)
        )
        for t in trades
    )
    returns = [t.return_pct for t in trades] if m.return_metrics_reliable else []
    currencies = {
        str(t.pnl_currency).upper() for t in trades if getattr(t, "pnl_currency", None)
    }
    missing_currency = sum(1 for t in trades if not getattr(t, "pnl_currency", None))
    if len(currencies) > 1 or (currencies and missing_currency):
        raise ValueError(
            "交易紀錄含不同或不明的 pnl_currency，金額不可直接相加；"
            "請先依同一帳戶/結算幣別分開分析。"
        )
    if len(currencies) == 1:
        m.pnl_currency = next(iter(currencies))
    elif not currencies:
        inferred = {
            currency
            for t in trades
            if (currency := infer_pnl_currency(t.symbol, t.market))
        }
        unresolved = sum(
            1 for t in trades if not infer_pnl_currency(t.symbol, t.market)
        )
        if len(inferred) > 1 or (inferred and unresolved):
            shown = "、".join(sorted(inferred)) or "未知"
            raise ValueError(
                f"交易紀錄未明示 pnl_currency，依商品推定出多種或不明幣別（{shown}；"
                f"不明 {unresolved} 筆），金額不可直接相加。"
            )
        m.currency_reliable = False
        if len(inferred) == 1:
            m.pnl_currency = next(iter(inferred))
            m.currency_note = (
                f"損益幣別僅依商品推定為 {m.pnl_currency}，未確認帳戶實際結算幣別；"
                "不可據此進入真錢驗證階段。"
            )
        else:
            m.currency_note = "損益幣別不明；不可和其他帳戶/市場金額合併。"

    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]

    m.wins = len(win_pnls)
    m.losses = len(loss_pnls)
    m.breakeven = m.total_trades - m.wins - m.losses

    m.win_rate = _safe_div(m.wins, m.total_trades)
    m.gross_profit = sum(win_pnls)
    m.gross_loss = abs(sum(loss_pnls))
    m.total_pnl = sum(pnls)
    m.total_fees = sum(t.fees for t in trades)

    m.avg_win = _safe_div(m.gross_profit, m.wins)
    m.avg_loss = _safe_div(m.gross_loss, m.losses)
    # 比率型指標走 _ratio:全勝樣本的盈虧比/獲利因子是「無虧損,不適用」,
    # 不是 0 —— 0 會被讀成「最差」,對全贏紀錄是顛倒黑白
    m.payoff_ratio = _ratio(m.avg_win, m.avg_loss)
    m.profit_factor = _ratio(m.gross_profit, m.gross_loss)

    # 期望值:每筆交易平均能賺/賠多少
    # E = 勝率 × 平均獲利 − 敗率 × 平均虧損
    loss_rate = _safe_div(m.losses, m.total_trades)
    m.expectancy = m.win_rate * m.avg_win - loss_rate * m.avg_loss

    # ── R-multiple:把每筆損益用「該筆的風險」標準化 ──
    # 以「平均虧損」作為 1R 的代理(沒有明確停損時的常見近似)。
    # 完全沒有虧損時 R 沒有定義(拿最小獲利當風險單位是張冠李戴),
    # 誠實留空,不編一個跨樣本不可比的數字。
    if m.avg_loss > 0:
        one_r = m.avg_loss
        m.r_multiples = [p / one_r for p in pnls]
        m.expectancy_r = _safe_div(sum(m.r_multiples), len(m.r_multiples))
    else:
        m.r_multiples = []
        m.expectancy_r = 0.0

    # largest_win 只從獲利單取、largest_loss 只從虧損單取(保留負號)——
    # 全勝時 min(pnls) 是「最小獲利」,填進 largest_loss 是欄位說謊
    m.largest_win = max(win_pnls, default=0.0)
    m.largest_loss = min(loss_pnls, default=0.0)
    # 最賺一筆佔總獲利比例:過高代表獲利集中在運氣,非穩定優勢
    m.top_trade_pnl_share = _safe_div(max(pnls, default=0.0), m.gross_profit)

    # ── 回撤:依時間順序累積權益,計算離峰值的最大跌幅 ──
    #
    # 回撤金額單純是「權益離峰值的最大跌幅」。但回撤『百分比』需要一個
    # 合理的基準資本 —— 若用「從零起算的累積損益峰值」當分母,峰值可能
    # 極小甚至接近 0,百分比會爆衝成上萬 %,毫無意義。
    #
    # 我們改用「估計初始資本」當分母:取所有交易中『最大單筆投入金額』
    # 作為帳戶資本規模的下限代理(實務上帳戶至少要撐得起最大那筆部位)。
    # 這讓回撤百分比落在「相對於帳戶規模」的合理區間。
    # 用 contract_value(= 價格 × 數量 × 契約乘數):期貨若漏乘乘數,
    # 資本基準會小 200 倍,回撤百分比整個失真。股票乘數為 1,結果不變。
    # 回撤百分比必須全程「因果」:分母與峰值都只能用「當下已知」的資訊。
    #  - 峰值用當下高水位(不可用最終峰值回算 —— 事後獲利會稀釋早期回撤)
    #  - 資本基準用「至今出現過的最大部位」逐筆更新(不可用整段紀錄的
    #    最大部位 —— 未來才放大的部位會把早期 90% 回撤稀釋成 0.009%)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0
    running_capital = 0.0
    if m.sequence_metrics_reliable:
        for t, p in zip(trades, pnls):
            running_capital = max(running_capital, t.contract_value)
            equity += p
            peak = max(peak, equity)
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
            denom_now = running_capital + peak  # 當下帳戶能動用的高水位
            if denom_now > 0:
                dd_pct = dd / denom_now
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
    else:
        # 具體原因已在排序守門處寫入 sequence_note。
        pass
    m.max_drawdown = max_dd
    m.max_drawdown_pct = max_dd_pct
    # pnl-only 資料(無進場價/數量)沒有資本基準:金額算得出、百分比算不出。
    # 標記不可靠,顯示層要說「無法計算」而不是印一個假的 0%。
    m.drawdown_pct_reliable = (
        m.sequence_metrics_reliable
        and m.return_metrics_reliable
        and running_capital > 0
        and not any(
            is_leveraged(t.market)
            or t.market == Market.CRYPTO
            or t.side == Side.SHORT
            for t in trades
        )
    )
    if (
        m.sequence_metrics_reliable
        and m.return_metrics_reliable
        and running_capital > 0
        and not m.drawdown_pct_reliable
    ):
        m.drawdown_note = (
            "紀錄含放空、期貨、選擇權、外匯或無法確認是否加槓桿的加密商品；"
            "契約名目價值不等於帳戶權益/保證金，因此帳戶回撤百分比無法計算。"
        )

    # ── 最長連續虧損 ──
    if m.sequence_metrics_reliable:
        streak = 0
        longest = 0
        for p in pnls:
            if p < 0:
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0
        m.max_consecutive_losses = longest

    # ── 夏普 / 索提諾(以每筆交易報酬率計算,非年化)──
    # 注意:這是「每筆交易」口徑,非年化。不同交易頻率的策略不可直接互比,
    # 僅用於同一份紀錄內的相對風險衡量。
    n = len(returns)
    mean_ret = _safe_div(sum(returns), n)
    if n > 1:
        var = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
        std = math.sqrt(var)
        # 零波動(全部報酬相同)的夏普是「不適用」,不是 0
        m.sharpe = _ratio(mean_ret, std)
        # 下行偏差:標準定義的分母是「全部樣本數」,不是只有下行筆數。
        # 用 len(downside) 當分母會系統性高估下行偏差、低估 Sortino,
        # 且方向錯誤(好策略下行少、分母小,反而被壓低)。以 0 為門檻,
        # 對每筆取 min(r, 0)^2,分母用 n-1 與夏普一致。
        dvar = sum(min(r, 0.0) ** 2 for r in returns) / (n - 1)
        dstd = math.sqrt(dvar)
        m.sortino = _ratio(mean_ret, dstd)
    if not m.return_metrics_reliable:
        m.return_note = (
            "至少一筆交易缺少可信的進場價、數量或契約乘數，"
            "或損益幣別與名目本金幣別不一致；"
            "報酬率、夏普、索提諾與回撤百分比無法計算。"
        )

    # ── 交易風格 ──
    timing_known = [
        t for t in trades
        if t.entry_time_known
        and t.exit_time_known
        and getattr(t, "time_basis_consistent", True)
    ]
    m.timing_known_trades = len(timing_known)
    m.timing_unknown_trades = len(trades) - len(timing_known)
    m.timing_data_complete = m.timing_unknown_trades == 0
    m.timing_metrics_available = bool(timing_known)
    holding = [t.holding_days for t in timing_known]
    holding = [days for days in holding if days is not None]
    m.avg_holding_days = _safe_div(sum(holding), len(holding))
    # 當沖判定統一用 Trade.is_day_trade(同一交易日),與成本估算口徑一致。
    # 分母也只能是雙時間已知的交易，不得把缺時間當成「非當沖」或「當沖」。
    intraday_count = sum(1 for t in timing_known if t.is_day_trade)
    m.intraday_ratio = _safe_div(intraday_count, len(timing_known))
    # 只有時間完整才能對整份紀錄下「以當沖為主」的判斷。
    m.is_mostly_intraday = (
        m.timing_data_complete
        and m.timing_metrics_available
        and m.intraday_ratio >= INTRADAY_RATIO_THRESHOLD
    )
    if not m.timing_data_complete:
        m.timing_note = (
            f"僅 {m.timing_known_trades}/{m.total_trades} 筆同時有進出場時間;"
            "平均持倉與當沖比例只來自已知子集,不對整體判定當沖風格。"
        )

    return m
