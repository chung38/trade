"""三市場交易成本模型。

很多人「帳面上」覺得自己賺錢,是因為他們沒把手續費、交易稅、滑價算進去。
這是賭徒最常見的自我欺騙之一。本模組在使用者沒提供 fees 時,
用各市場的「真實成本」自動估算,讓盈虧回到誠實的數字。

注意:這些是常見的預設值,使用者應依自己的券商/交易所實際費率調整。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Market, Side


@dataclass
class CostModel:
    """單一市場的成本參數。

    支援兩種手續費結構:
      - 比例式(股票、加密貨幣):成交金額 × commission_rate
      - 每口固定(期貨、選擇權):口數 × commission_per_unit
    兩者可並存,取「比例式 + 固定式」之和,再與 commission_min 取大者。

    Attributes:
        commission_rate:     手續費率(以契約價值計)
        commission_min:      最低手續費(每筆)
        commission_per_unit: 每口/每單位的固定手續費(期貨常見)
        tax_rate:            交易稅率
        tax_both_sides:      稅是否買賣雙邊都收(期貨/選擇權為 True;股票只在賣出收)
        slippage_rate:       預估滑價率(每邊)
    """

    commission_rate: float
    commission_min: float
    tax_rate: float
    slippage_rate: float
    commission_per_unit: float = 0.0
    tax_both_sides: bool = False
    # 賣出端「按股數」計的規費(美股 FINRA TAF 結構):每股費率 + 每筆上限。
    # 0 = 不適用。
    sell_per_share_fee: float = 0.0
    sell_per_share_fee_cap: float = 0.0

    def estimate(
        self,
        price: float,
        quantity: float,
        *,
        is_sell: bool,
        contract_multiplier: float = 1.0,
    ) -> float:
        """估算「單邊」的成本(進場一次、出場一次各算一次)。

        稅基一律用「契約價值」= 價格 × 乘數 × 數量。
        對股票乘數為 1,結果與舊版逐位相同(向後相容)。
        """
        value = abs(price * quantity * contract_multiplier)
        commission = max(
            value * self.commission_rate + self.commission_per_unit * abs(quantity),
            self.commission_min,
        )
        charge_tax = is_sell or self.tax_both_sides
        tax = value * self.tax_rate if charge_tax else 0.0
        # 賣出端按股數的規費(FINRA TAF):每股 × 股數,有每筆上限。
        # 低價例外:成交價低於每股費率時 FINRA 不收 TAF。
        per_share = 0.0
        if is_sell and self.sell_per_share_fee > 0 and price >= self.sell_per_share_fee:
            per_share = self.sell_per_share_fee * abs(quantity)
            if self.sell_per_share_fee_cap > 0:
                per_share = min(per_share, self.sell_per_share_fee_cap)
        slippage = value * self.slippage_rate
        return commission + tax + per_share + slippage


# 各市場的預設成本模型(2026 年常見值,僅供估算)。
DEFAULT_COST_MODELS: dict[Market, CostModel] = {
    # 台股:手續費 0.1425%(常見打折後約 0.06%),賣出證交稅 0.3%(當沖 0.15%)
    Market.TW_STOCK: CostModel(
        commission_rate=0.001425,
        commission_min=20.0,      # 多數券商最低 20 元
        tax_rate=0.003,
        slippage_rate=0.0005,
    ),
    # 美股:多數券商零佣金,但有 SEC/FINRA 規費與點差滑價。
    # 費率 as_of 2026-04(法定費率每年調整,估算值,非精算):
    #   SEC Section 31:賣出 $20.60 / 百萬美元(FY2026,2026-04-04 生效)
    #   FINRA TAF:賣出每股 $0.000195,每筆上限 $9.79
    Market.US_STOCK: CostModel(
        commission_rate=0.0,
        commission_min=0.0,
        tax_rate=0.0000206,
        slippage_rate=0.0005,
        sell_per_share_fee=0.000195,
        sell_per_share_fee_cap=9.79,
    ),
    # 加密貨幣:交易所現貨手續費約 0.1%(雙邊),滑價在小幣上可能很大
    Market.CRYPTO: CostModel(
        commission_rate=0.001,
        commission_min=0.0,
        tax_rate=0.0,
        slippage_rate=0.0010,
    ),
    # 台股 ETF:證交稅 0.1%(股票型)。債券 ETF 至 2026-12-31 暫停課徵,
    # 但無法從代號辨識型別 —— 一律用股票型 0.1% 保守估(寧可估貴),
    # 並在 markets.uncovered_cost_warnings 對使用者揭露這件事。
    Market.TW_ETF: CostModel(
        commission_rate=0.001425,
        commission_min=20.0,
        tax_rate=0.001,
        slippage_rate=0.0005,
    ),
    # 台指期:每口固定手續費;期交稅十萬分之二,課「契約價值」且買賣雙邊都收
    Market.TW_FUTURES: CostModel(
        commission_rate=0.0,
        commission_min=0.0,
        commission_per_unit=20.0,   # 每口約 20 元(各券商不同)
        tax_rate=0.00002,           # 十萬分之二
        tax_both_sides=True,
        # 滑價約 1 檔(台指期 1 點);以指數 23000 計約為 1/23000 ≈ 0.00004
        slippage_rate=0.00005,
    ),
    # 台指選擇權:稅率千分之一,課「權利金」(價格 × 乘數),買賣雙邊都收
    Market.TW_OPTIONS: CostModel(
        commission_rate=0.0,
        commission_min=0.0,
        commission_per_unit=15.0,
        tax_rate=0.001,
        tax_both_sides=True,
        slippage_rate=0.0010,
    ),
    # 外匯:成本主要是點差(以滑價近似);隔夜利息 swap 未建模(見 markets.py 警語)
    Market.FOREX: CostModel(
        commission_rate=0.0,
        commission_min=0.0,
        tax_rate=0.0,
        slippage_rate=0.0002,       # 約 2 pip 的點差近似
    ),
    Market.UNKNOWN: CostModel(
        commission_rate=0.001,
        commission_min=0.0,
        tax_rate=0.0,
        slippage_rate=0.0005,
    ),
}


# 台股現股當沖證交稅減半(0.3% → 0.15%)。此優惠自 2017 年起多次延長,
# 截至 2026 年仍有效。當沖者的成本若用全額稅率會被高估一倍,可能讓原本
# 打平的當沖被誤判為虧損 / 負期望 → 誤觸賭博裁決,與工具宗旨衝突。
TW_DAY_TRADE_TAX_RATE = 0.0015


def estimate_round_trip_cost(
    market: Market,
    side: Side,
    entry_price: float,
    exit_price: float,
    quantity: float,
    model: CostModel | None = None,
    *,
    is_day_trade: bool = False,
    contract_multiplier: float = 1.0,
) -> float:
    """估算一筆「完整來回」交易的總成本(進場 + 出場)。

    做多:買進(進場,不收稅)→ 賣出(出場,收稅)
    做空:賣出(進場,收稅)→ 買回(出場,不收稅)
    期貨 / 選擇權:買賣雙邊都收期交稅(tax_both_sides)。

    Args:
        is_day_trade:        是否為當沖(進出場同一交易日)。台股當沖證交稅減半。
        contract_multiplier: 契約乘數。股票/加密貨幣為 1.0(結果與舊版相同)。
    """
    m = model or DEFAULT_COST_MODELS.get(market, DEFAULT_COST_MODELS[Market.UNKNOWN])

    # 台股當沖:證交稅減半。複製一份模型,只調整稅率,不動其他參數。
    if is_day_trade and market == Market.TW_STOCK and m.tax_rate > TW_DAY_TRADE_TAX_RATE:
        # 用 dataclasses.replace 只改稅率:手寫欄位複製在新增欄位時會
        # 靜默清零(sell_per_share_fee 就曾被這樣吃掉)
        import dataclasses
        m = dataclasses.replace(m, tax_rate=TW_DAY_TRADE_TAX_RATE)

    mult = contract_multiplier
    if side == Side.LONG:
        entry_cost = m.estimate(entry_price, quantity, is_sell=False, contract_multiplier=mult)
        exit_cost = m.estimate(exit_price, quantity, is_sell=True, contract_multiplier=mult)
    else:  # SHORT
        entry_cost = m.estimate(entry_price, quantity, is_sell=True, contract_multiplier=mult)
        exit_cost = m.estimate(exit_price, quantity, is_sell=False, contract_multiplier=mult)

    return entry_cost + exit_cost
