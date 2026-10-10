"""市場與商品規格 — 契約乘數、辨識規則、槓桿標註。

為什麼需要這個模組:

    台指期一口的契約乘數是 200(小台 50、選擇權 50)。如果不乘這個數字,
    一口台指期漲 10 點,損益會被算成 10 元而不是 2000 元 —— **少算 200 倍**。
    這個錯誤會靜默地汙染下游所有東西:期望值、回撤、顯著性檢定、趨勢、夏普值。

    乘數無法從代號或價位可靠反推(同樣是「台指」,大台 200、小台 50;
    不同券商、不同帳戶別也可能不同)。所以本模組採**白名單**:
    查得到就用,查不到就標記 multiplier_unknown 並拒絕自動估算成本,
    而不是猜一個數字。寧可不算,不可算錯。

誠實聲明:所有費率都是 2026 年的常見估計值,會隨主管機關與券商調整。
請以你的實際對帳單為準。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Market

#: 本檔費率的最後查核日期(提醒使用者資訊會過時)
LAST_REVIEWED = "2026-07"


@dataclass(frozen=True)
class MarketSpec:
    """單一市場的商品規格。"""

    market: Market
    unit_name: str            # 數量的單位名稱(股 / 口 / 張 / 幣)
    is_leveraged: bool        # 是否為槓桿商品(影響 return_pct 與夏普的可比性)
    default_multiplier: float # 找不到白名單時的預設乘數
    note: str = ""


MARKET_SPECS: dict[Market, MarketSpec] = {
    Market.TW_STOCK: MarketSpec(Market.TW_STOCK, "股", False, 1.0),
    Market.TW_ETF: MarketSpec(Market.TW_ETF, "股", False, 1.0,
                              "股票型 ETF 證交稅 0.1%(非 0.3%)"),
    Market.US_STOCK: MarketSpec(Market.US_STOCK, "股", False, 1.0),
    Market.CRYPTO: MarketSpec(Market.CRYPTO, "幣", False, 1.0,
                              "永續合約有資金費率,本工具未建模"),
    Market.TW_FUTURES: MarketSpec(Market.TW_FUTURES, "口", True, 1.0,
                                  "乘數必須查白名單;查不到不可自動估成本"),
    Market.TW_OPTIONS: MarketSpec(Market.TW_OPTIONS, "口", True, 1.0,
                                  "賣方風險左尾極厚,樣本內可能完全看不到爆倉"),
    Market.FOREX: MarketSpec(
        Market.FOREX,
        "基礎貨幣單位",
        True,
        1.0,
        "數量必須填 base units，不能把手數直接當數量；"
        "隔夜利息(swap)可正可負,本工具未建模",
    ),
    Market.UNKNOWN: MarketSpec(Market.UNKNOWN, "單位", False, 1.0),
}


# ── 契約乘數白名單 ────────────────────────────────────────────
# 只放我們有把握的。查不到就是查不到,不猜。
SYMBOL_MULTIPLIERS: dict[str, float] = {
    # 台灣期貨(台指期系列)
    "TXF": 200.0,   # 大台
    "MXF": 50.0,    # 小台
    "MX1": 50.0,    # 週小台 W1
    "MX2": 50.0,    # 週小台 W2
    "MX4": 50.0,    # 週小台 W4
    "MX5": 50.0,    # 週小台 W5
    "TMF": 10.0,    # 微台
    "EXF": 4000.0,  # 電子期
    "FXF": 1000.0,  # 金融期
    # 台灣選擇權(權利金點數 × 50)
    "TXO": 50.0,
    "TX1": 50.0,    # 週三到期 W1
    "TX2": 50.0,    # 週三到期 W2
    "TX4": 50.0,    # 週三到期 W4
    "TX5": 50.0,    # 週三到期 W5
    "TXU": 50.0,    # 週五到期 F1
    "TXV": 50.0,    # 週五到期 F2
    "TXX": 50.0,    # 週五到期 F3
    "TXY": 50.0,    # 週五到期 F4
    "TXZ": 50.0,    # 週五到期 F5
}

_TW_OPTION_PREFIXES = frozenset({
    "TXO", "TX1", "TX2", "TX4", "TX5", "TXU", "TXV", "TXX", "TXY", "TXZ",
})

# 台灣期貨/選擇權代號 = 白名單前綴 + 契約月份碼(必須含數字)。
# 例:TXFG5(月份碼 G + 年碼 5)、TXF202607(年月)、TXO18000G5(履約價 + 月份碼)。
# 「必須含數字」是關鍵防線:TMF / FXF / TXO 都是真實美股代號(純字母),
# 裸前綴比對會把它們誤判成台期,損益錯 10~4000 倍。
_TW_DERIV_PATTERN = re.compile(
    r"^(" + "|".join(sorted(SYMBOL_MULTIPLIERS, key=len, reverse=True)) + r")(?=[A-Z0-9]*\d)"
)

# ISO 4217 主要貨幣(用於辨識外匯代號如 EURUSD)
ISO_CCY = frozenset({
    "USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF",
    "CNY", "CNH", "HKD", "SGD", "TWD", "KRW", "SEK", "NOK",
    "MXN", "ZAR", "TRY", "PLN", "AED", "ARS", "BGN", "BHD",
    "BRL", "CLP", "COP", "CZK", "DKK", "EGP", "HUF", "IDR",
    "ILS", "INR", "ISK", "JOD", "KES", "KWD", "MAD", "MYR",
    "NGN", "OMR", "PHP", "PKR", "QAR", "RON", "RUB", "SAR",
    "THB", "VND", "XAG", "XAU",
})

# 加密貨幣的計價幣(用於區分 BTCUSDT 這種代號)
_CRYPTO_QUOTE = re.compile(r"(USDT|USDC|BUSD|DAI)$", re.IGNORECASE)
# 主流幣「基底」只有在後面接分隔符或明確計價幣時才算加密貨幣 ——
# 裸前綴比對是真實 bug:SOL(NYSE 真實代號)、ETHA(iShares 以太幣 ETF)、
# SOLAR 等美股會被劫持成 crypto,套用完全錯誤的成本模型。
_CRYPTO_BASE = re.compile(
    r"^(BTC|ETH|SOL|XRP|DOGE|ADA|BNB)(?=[-/_]|(USD|EUR|JPY|GBP|TWD|KRW|BTC|ETH|BNB)$)",
    re.IGNORECASE,
)
# 裸幣名(無計價幣)與證券代號空間衝突(SOL/ADA 都可能是美股代號),
# 模稜兩可 → UNKNOWN,要用 market_hint 明確指定。寧可不判,不可錯判。
_BARE_COINS = frozenset({"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "BNB"})

_FOREX_PAIR = re.compile(r"^([A-Z]{3})([A-Z]{3})$")
_TW_ETF = re.compile(r"^00\d{2,4}[A-Z]?$")      # 0050, 0056, 00878...
_TW_STOCK = re.compile(r"^\d{4,6}[A-Z]?$")
# 美股類別股可含一碼 dot/hyphen 後綴(例如 BRK.B；部分資料源輸出 BRK-B)。
# 加密貨幣的明確分隔格式已在前面先判斷,不會被這條較寬的規則劫持。
_US_STOCK = re.compile(r"^[A-Z]{1,5}(?:[.-][A-Z])?$")


def contract_multiplier(symbol: str) -> tuple[float, bool]:
    """查契約乘數。

    Returns:
        (乘數, 是否查得到)。查不到時回 (1.0, False) —— 呼叫端應標記
        multiplier_unknown 並拒絕對槓桿商品自動估算成本。
    """
    s = str(symbol).strip().upper()
    m = _TW_DERIV_PATTERN.match(s)
    if m:
        return SYMBOL_MULTIPLIERS[m.group(1)], True
    return 1.0, False


def infer_market(symbol: str, hint: Market | None = None) -> Market:
    """從標的代號推斷市場別。

    辨識順序很重要 —— 修正了一個真實 bug:舊版的加密貨幣規則是
    「以 USD 結尾」,會把 EURUSD / GBPUSD / AUDUSD 這些**外匯**代號
    誤判成加密貨幣,導致套用完全錯誤的成本模型。

    模稜兩可時一律回 UNKNOWN(寧可不判,不可錯判)。
    """
    if hint and hint != Market.UNKNOWN:
        return hint
    s = str(symbol).strip().upper()
    if not s:
        return Market.UNKNOWN

    # 1. 期貨 / 選擇權:白名單前綴 + 必須帶契約月份碼。
    #    不可用裸 startswith —— TMF(美股 3 倍做多公債 ETF)、FXF(瑞郎 ETF)、
    #    TXO(TXO Partners, NYSE)都是**真實存在的美股代號**,裸前綴比對會把
    #    美股使用者的損益放大 10~4000 倍。真實台期代號一定帶月份碼
    #    (如 TXFG5、TXF202607),裸三字母代號一律不視為期貨。
    m_deriv = _TW_DERIV_PATTERN.match(s)
    if m_deriv:
        prefix = m_deriv.group(1)
        return Market.TW_OPTIONS if prefix in _TW_OPTION_PREFIXES else Market.TW_FUTURES

    # 2. 外匯:六碼且前三後三都是 ISO 貨幣(必須早於加密貨幣判斷)
    # 券商常輸出 EUR/USD、EUR-USD、EUR_USD。只在去掉單一分隔符後
    # 兩側都是已知法幣才視為 FX；BTC-USD 的 BTC 不在 ISO 集合，仍走加密。
    forex_s = re.sub(r"[/_-]", "", s)
    m = _FOREX_PAIR.match(forex_s)
    if m and m.group(1) in ISO_CCY and m.group(2) in ISO_CCY:
        return Market.FOREX

    # 3. 加密貨幣:明確的計價幣後綴,或「主流幣基底 + 分隔符/計價幣」
    if _CRYPTO_QUOTE.search(s) or _CRYPTO_BASE.match(s):
        return Market.CRYPTO

    # 3.5 裸幣名(BTC、SOL…):與美股代號空間衝突,誠實回 UNKNOWN
    if s in _BARE_COINS:
        return Market.UNKNOWN

    # 4. 台股 ETF(00 開頭)→ 5. 台股 → 6. 美股
    if _TW_ETF.match(s):
        return Market.TW_ETF
    if _TW_STOCK.match(s):
        return Market.TW_STOCK
    if _US_STOCK.match(s):
        return Market.US_STOCK

    return Market.UNKNOWN


def infer_pnl_currency(symbol: str, market: Market | None = None) -> str | None:
    """依商品推定常見損益幣別；使用者明示的帳戶幣別仍應優先。

    這只用來阻止 TWD、USD、JPY 等金額被靜默相加，不做匯率換算。
    """

    s = str(symbol).strip().upper().replace("/", "").replace("-", "").replace("_", "")
    mkt = market or infer_market(symbol)
    if mkt in (Market.TW_STOCK, Market.TW_ETF, Market.TW_FUTURES, Market.TW_OPTIONS):
        return "TWD"
    if mkt == Market.US_STOCK:
        return "USD"
    if mkt == Market.FOREX and len(s) == 6:
        return s[-3:]
    if mkt == Market.CRYPTO:
        for quote in ("USDT", "USDC", "BUSD", "DAI", "USD", "TWD", "EUR", "JPY", "BTC", "ETH", "BNB"):
            if s.endswith(quote) and len(s) > len(quote):
                return quote
    return None


def is_leveraged(market: Market) -> bool:
    """槓桿商品:報酬率與夏普值不可跨商品比較(母體定義不同)。"""
    spec = MARKET_SPECS.get(market)
    return bool(spec and spec.is_leveraged)


def unit_name(market: Market) -> str:
    spec = MARKET_SPECS.get(market)
    return spec.unit_name if spec else "單位"


def uncovered_cost_warnings(market: Market) -> list[str]:
    """本工具「未涵蓋」的成本 —— 必須誠實告訴使用者。"""
    out: list[str] = []
    if market == Market.FOREX:
        out.append(
            "外匯的隔夜利息(swap)可正可負、逐日浮動、各券商加價不一,"
            "本工具未建模。請以對帳單的實際費用填入 fees 欄位。"
        )
    if market == Market.CRYPTO:
        out.append(
            "加密貨幣永續合約的資金費率(funding rate)未建模。"
            "若你做的是永續合約,請把資金費用計入 fees。"
        )
    if market == Market.TW_ETF:
        out.append(
            "一般債券 ETF(不含槓桿型/反向型)至 2026-12-31 暫停課徵證交稅,"
            "但本工具無法從代號辨識 ETF 型別,一律用股票型 0.1% 保守估算 —— "
            "若你交易的是適用免稅的債券 ETF,賣出稅被高估,請在 fees 欄位填實際費用。"
        )
    if market in (Market.TW_FUTURES, Market.TW_OPTIONS):
        out.append(
            "期貨/選擇權的保證金追繳、強制平倉未建模;報酬率以契約價值為母體計算,"
            "與以保證金為母體的算法差距可達數十倍,不可與股票的報酬率直接比較。"
        )
    if market == Market.TW_OPTIONS:
        out.append(
            "⚠️ 選擇權賣方的虧損左尾極厚:歷史樣本裡「剛好沒發生爆倉」時,"
            "統計檢定會誤判為穩定獲利。顯著為正的結果對賣方策略特別不可信。"
        )
    return out
