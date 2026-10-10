"""券商交易截圖的保守式 OCR 文字解析。

這個模組刻意分成兩層：

* :func:`parse_ocr_text` 只吃文字，核心零依賴、容易測試。
* :func:`parse_screenshot_image` 可選擇注入 OCR 函式；未注入時才嘗試
  ``Pillow + pytesseract``，因此不會把圖片套件變成核心依賴。

OCR 最危險的錯誤不是「讀不到」，而是把價格、正負號或買賣方向讀錯後仍
自動送進績效分析。因此本模組只把高信心且沒有衝突的候選放進
``selected_fields``。低信心值仍保留在 ``candidates`` 供畫面逐項確認，但
絕不由 :meth:`ScreenshotParseResult.value` 回傳。

策略字眼（突破、均線、RSI 等）只標記為「文字線索」，不能用來認證策略
存在，更不能據此宣稱有統計優勢。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_ACCEPTANCE_THRESHOLD = 0.80

# 對外欄位名稱刻意和 loader 的標準欄位一致；額外的 action / buy_price /
# sell_price / fill_price / trade_time 是「尚不能安全對應」的中介欄位。
PRIMARY_FIELDS = (
    "symbol",
    "side",
    "entry_price",
    "exit_price",
    "quantity",
    "entry_time",
    "exit_time",
    "pnl",
    "stop_loss",
    "take_profit",
)

FIELD_LABELS = {
    "symbol": "標的",
    "side": "方向",
    "entry_price": "進場價",
    "exit_price": "出場價",
    "quantity": "數量",
    "entry_time": "進場時間",
    "exit_time": "出場時間",
    "pnl": "已實現損益",
    "stop_loss": "停損",
    "take_profit": "停利",
    "action": "交易動作",
    "buy_price": "買進價",
    "sell_price": "賣出價",
    "fill_price": "單次成交價",
    "trade_time": "單次成交時間",
    "unrealized_pnl": "未實現損益",
}

STRATEGY_NOTICE = (
    "技術/策略線索只代表 OCR 文字出現相關字樣；不代表使用者確實依此交易，"
    "也不代表該方法具有可重複的統計優勢。"
)


@dataclass(frozen=True)
class FieldCandidate:
    """一個有來源證據的欄位候選。

    ``candidate_value`` 可能是低信心值；下游不得直接拿它分析。安全入口是
    :meth:`ScreenshotParseResult.value` 或 ``selected_fields``。
    """

    field: str
    candidate_value: str | float
    raw_value: str
    confidence: float
    evidence: str
    line_number: int | None
    basis: str
    unit: str | None = None
    derived: bool = False

    @property
    def label(self) -> str:
        return FIELD_LABELS.get(self.field, self.field)


@dataclass(frozen=True)
class StrategyClue:
    """從畫面文字找到的技術/策略字眼；僅供描述與人工確認。"""

    code: str
    label: str
    evidence: str
    line_number: int
    confidence: float


@dataclass
class ScreenshotParseResult:
    """保守式解析結果。

    ``selected_fields`` 只包含通過門檻、且同欄沒有相近衝突的候選。
    ``candidates`` 則保留所有讀到的可能值，方便新手表單顯示原文覆核。
    """

    source_text: str
    candidates: tuple[FieldCandidate, ...]
    selected_fields: dict[str, FieldCandidate]
    strategy_clues: tuple[StrategyClue, ...]
    warnings: tuple[str, ...] = ()
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD
    source: str = "ocr_text"

    def value(self, field_name: str) -> str | float | None:
        """只回傳已安全選取的值；低信心或衝突候選一律回 ``None``。"""

        candidate = self.selected_fields.get(field_name)
        return candidate.candidate_value if candidate is not None else None

    def candidates_for(self, field_name: str) -> tuple[FieldCandidate, ...]:
        """取得某欄所有候選（含低信心），依信心由高至低排序。"""

        return tuple(
            sorted(
                (candidate for candidate in self.candidates if candidate.field == field_name),
                key=lambda candidate: candidate.confidence,
                reverse=True,
            )
        )

    def safe_values(self, *, primary_only: bool = True) -> dict[str, str | float]:
        """回傳可自動帶入表單的安全值，不包含任何待確認候選。"""

        allowed = set(PRIMARY_FIELDS) if primary_only else None
        return {
            name: candidate.candidate_value
            for name, candidate in self.selected_fields.items()
            if allowed is None or name in allowed
        }

    @property
    def needs_manual_review(self) -> bool:
        """截圖 OCR 永遠需要人工逐欄對照；保留欄名供舊整合相容。"""

        return True

    @property
    def requires_human_review(self) -> bool:
        """明確語意的新欄位：任何截圖/OCR 結果都不可免人工覆核。"""

        return True

    @property
    def missing_required_fields(self) -> tuple[str, ...]:
        """尚未安全取得的建檔核心欄位，與人工覆核需求分開表達。"""

        required = {"symbol", "side", "entry_price", "exit_price", "quantity"}
        return tuple(sorted(required - self.selected_fields.keys()))


class OCRUnavailableError(RuntimeError):
    """沒有可用 OCR 引擎，或 OCR 引擎無法讀取圖片。"""


@dataclass(frozen=True)
class _NumberRule:
    field: str
    labels: tuple[str, ...]
    confidence: float
    basis: str
    unit_kind: str
    sign_hint: int | None = None


_NUMBER_TOKEN = re.compile(
    r"(?P<number>[\(（]?\s*[+\-−－]?\s*"
    r"(?:(?:NT|US)?\$|TWD|NTD|USD|USDT|USDC|BUSD|JPY|RMB|CNY|CNH|HKD|"
    r"EUR|GBP|CHF|CAD|AUD|NZD|SGD|KRW|￥|¥)?\s*"
    r"\d(?:[\d,，]*)(?:\.\d+)?\s*%?\s*[\)）]?)",
    re.IGNORECASE,
)

_TIME_TOKEN = re.compile(
    r"(?P<time>"
    r"民國\s*\d{2,3}\s*[年/.-]\s*\d{1,2}\s*[月/.-]\s*\d{1,2}\s*日?"
    r"(?:[ T]*\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{4}\s*[年/.-]\s*\d{1,2}\s*[月/.-]\s*\d{1,2}\s*日?"
    r"(?:[ T]*\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{2,3}[/.\-]\d{1,2}[/.\-]\d{1,2}"
    r"(?:[ T]*\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}[/.\-]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r")",
    re.IGNORECASE,
)

_NUMBER_RULES = (
    _NumberRule(
        "entry_price",
        (
            "進場成交均價", "進場均價", "進場價", "開倉成交價", "開倉價",
            "AVG ENTRY PRICE", "ENTRY PRICE", "OPENING PRICE", "OPEN PRICE",
        ),
        0.98,
        "畫面直接標示進場/開倉價格",
        "price",
    ),
    _NumberRule(
        "exit_price",
        (
            "出場成交均價", "出場均價", "出場價", "平倉成交價", "平倉價",
            "AVG EXIT PRICE", "EXIT PRICE", "CLOSING PRICE", "CLOSE PRICE",
        ),
        0.98,
        "畫面直接標示出場/平倉價格",
        "price",
    ),
    _NumberRule(
        "buy_price",
        (
            "買進成交均價", "買入成交均價", "買進均價", "買入均價", "買進價", "買入價",
            "AVG BUY PRICE", "BUY PRICE",
        ),
        0.95,
        "畫面標示買進價格；尚不能單憑買進判定是進場或回補",
        "price",
    ),
    _NumberRule(
        "sell_price",
        ("賣出成交均價", "賣出均價", "賣出價", "賣價", "AVG SELL PRICE", "SELL PRICE"),
        0.95,
        "畫面標示賣出價格；尚不能單憑賣出判定是出場或放空",
        "price",
    ),
    _NumberRule(
        "quantity",
        (
            "成交數量", "成交股數", "交易數量", "委託數量", "數量", "股數", "張數", "口數",
            "QUANTITY", "QTY", "SHARES", "CONTRACTS", "SIZE", "AMOUNT",
        ),
        0.96,
        "畫面直接標示數量",
        "quantity",
    ),
    _NumberRule(
        "stop_loss",
        ("停損價格", "停損價", "停損", "止損價格", "止損價", "止損", "STOP LOSS", "SL"),
        0.97,
        "畫面直接標示停損",
        "risk_level",
    ),
    _NumberRule(
        "take_profit",
        ("停利價格", "停利價", "停利", "止盈價格", "止盈價", "止盈", "TAKE PROFIT", "TP"),
        0.97,
        "畫面直接標示停利",
        "risk_level",
    ),
    _NumberRule(
        "pnl",
        ("已實現損益", "已實現盈虧", "REALIZED PNL", "REALIZED P&L"),
        0.98,
        "畫面直接標示已實現損益",
        "pnl",
    ),
    _NumberRule(
        "unrealized_pnl",
        (
            "未實現損益", "未實現盈虧", "浮動損益", "浮動盈虧", "未平倉損益",
            "未平倉盈虧", "持倉損益", "持倉盈虧", "帳面損益", "帳面盈虧",
            "UNREALIZED PNL", "UNREALIZED P&L", "FLOATING PNL", "FLOATING P&L",
            "OPEN PNL", "OPEN P&L", "POSITION PNL", "POSITION P&L",
        ),
        0.98,
        "畫面標示未實現/浮動損益，不能當成已平倉損益",
        "pnl",
    ),
    _NumberRule(
        "pnl",
        ("虧損", "虧", "LOSS"),
        0.92,
        "『虧損』文字明示負號",
        "pnl",
        sign_hint=-1,
    ),
    _NumberRule(
        "pnl",
        ("獲利", "盈利", "PROFIT"),
        0.92,
        "『獲利』文字明示正號",
        "pnl",
        sign_hint=1,
    ),
    _NumberRule(
        "pnl",
        ("總損益", "損益金額", "淨損益", "損益", "盈虧", "PNL", "P&L"),
        0.90,
        "畫面標示損益，但 OCR 文字必須保留正負號才能安全採用",
        "pnl",
    ),
    _NumberRule(
        "fill_price",
        ("成交均價", "平均成交價", "成交價", "委託價", "PRICE"),
        0.92,
        "畫面只標示單次成交價，尚不能判定是進場或出場",
        "price",
    ),
)

_TIME_RULES = (
    (
        "entry_time",
        (
            "進場成交時間", "進場時間", "開倉時間", "買進開倉時間",
            "ENTRY TIME", "OPEN TIME", "OPENING TIME",
        ),
        0.98,
        "畫面直接標示進場/開倉時間",
    ),
    (
        "exit_time",
        (
            "出場成交時間", "出場時間", "平倉時間", "賣出平倉時間",
            "EXIT TIME", "CLOSE TIME", "CLOSING TIME",
        ),
        0.98,
        "畫面直接標示出場/平倉時間",
    ),
    (
        "trade_time",
        ("成交日期時間", "成交時間", "交易時間", "成交日期", "交易日期", "TIME"),
        0.94,
        "畫面只標示單次成交時間，尚不能判定是進場或出場",
    ),
)

_STRATEGY_PATTERNS = (
    ("breakout", "突破", re.compile(r"突破|breakout|創(?:新)?高|過前高|破底", re.I)),
    (
        "moving_average",
        "均線",
        re.compile(
            r"均線|(?:^|[^A-Z])(?:SMA|EMA|MA)\s*\d{1,3}(?:$|[^A-Z0-9])|"
            r"\d{1,3}\s*(?:日|週|月)線",
            re.I,
        ),
    ),
    (
        "cross_signal",
        "交叉訊號",
        re.compile(r"黃金交叉|死亡交叉|golden cross|death cross", re.I),
    ),
    ("rsi", "RSI", re.compile(r"(?:^|[^A-Z])RSI\s*(?:\(?\d{1,3}(?:\.\d+)?\)?)?", re.I)),
    ("macd", "MACD", re.compile(r"(?:^|[^A-Z])MACD(?:$|[^A-Z])", re.I)),
    (
        "stochastic",
        "KD/隨機指標",
        re.compile(r"(?:^|[^A-Z])KD(?:J)?(?:$|[^A-Z])|隨機指標|stochastic", re.I),
    ),
    ("bollinger", "布林通道", re.compile(r"布林|bollinger|(?:^|[^A-Z])BBANDS?(?:$|[^A-Z])", re.I)),
    ("volume", "成交量", re.compile(r"成交量|爆量|量增|量縮|volume", re.I)),
    ("support_resistance", "支撐/壓力", re.compile(r"支撐|壓力|阻力|support|resistance", re.I)),
    ("vwap", "VWAP", re.compile(r"(?:^|[^A-Z])VWAP(?:$|[^A-Z])", re.I)),
    ("fibonacci", "費波那契", re.compile(r"費波|斐波|fibonacci", re.I)),
    ("gap", "跳空缺口", re.compile(r"跳空|缺口|gap", re.I)),
    ("candlestick", "K線型態", re.compile(r"K線|紅K|黑K|吞噬|十字線|candlestick", re.I)),
    ("trendline", "趨勢線", re.compile(r"趨勢線|上升趨勢|下降趨勢|trendline", re.I)),
)


def _normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = line.replace("−", "-").replace("－", "-")
    return re.sub(r"[\t ]+", " ", line).strip()


def _meaningful_lines(text: str) -> list[tuple[int, str]]:
    return [
        (line_no, normalized)
        for line_no, raw in enumerate(text.splitlines(), 1)
        if (normalized := _normalize_line(raw))
    ]


def _flex_label(label: str) -> str:
    """允許 OCR 把中文標籤拆成「進 場 價」或英文空白不穩定。"""

    normalized = unicodedata.normalize("NFKC", label).strip()
    compact = re.sub(r"\s+", "", normalized)
    if compact and all("\u4e00" <= char <= "\u9fff" for char in compact):
        return r"\s*".join(re.escape(char) for char in compact)
    words = re.split(r"\s+", normalized)
    return r"\s*".join(re.escape(word) for word in words)


def _label_pattern(labels: Iterable[str]) -> re.Pattern[str]:
    alternatives = sorted((_flex_label(label) for label in labels), key=len, reverse=True)
    return re.compile(
        r"(?<![A-Z0-9\u4e00-\u9fff])(?:" + "|".join(alternatives) + r")"
        r"(?![A-Z\u4e00-\u9fff])\s*[:：=]?\s*",
        re.IGNORECASE,
    )


def _value_fragment(
    lines: list[tuple[int, str]],
    index: int,
    label_match: re.Match[str],
) -> tuple[str, str, int]:
    """取得標籤後文字；若標籤獨占一行，再看下一個非空 OCR 行。"""

    line_no, line = lines[index]
    fragment = line[label_match.end():].strip(" |｜")
    if fragment:
        return fragment, line, line_no
    if index + 1 < len(lines):
        next_no, next_line = lines[index + 1]
        return next_line, f"{line} ↵ {next_line}", line_no
    return "", line, line_no


def _parse_number(raw: str) -> tuple[float | None, bool, bool]:
    """回傳 (數值, 是否有明示正負號, 是否為百分比)。"""

    normalized = unicodedata.normalize("NFKC", raw).strip()
    is_percent = "%" in normalized
    parenthesized = normalized.startswith("(") and normalized.endswith(")")
    has_sign = bool(re.search(r"[+\-]", normalized)) or parenthesized
    cleaned = re.sub(
        r"(?:NT|US)?\$|TWD|NTD|USD|USDT|USDC|BUSD|JPY|RMB|CNY|CNH|HKD|"
        r"EUR|GBP|CHF|CAD|AUD|NZD|SGD|KRW|￥|¥|[%\s,]",
        "",
        normalized,
        flags=re.I,
    )
    cleaned = cleaned.replace("(", "").replace(")", "")
    try:
        value = float(cleaned)
    except (TypeError, ValueError):
        return None, has_sign, is_percent
    if parenthesized:
        value = -abs(value)
    return value, has_sign, is_percent


def _number_unit(raw: str, tail: str, unit_kind: str, evidence: str = "") -> str:
    if "%" in raw:
        return "percent"
    if unit_kind == "quantity":
        unit_match = re.match(r"\s*(張|股|口|枚|幣|單位|shares?|lots?|contracts?)", tail, re.I)
        unit = unit_match.group(1).lower() if unit_match else ""
        if unit == "張" or unit.startswith("lot"):
            return "lot"
        if unit == "股" or unit.startswith("share"):
            return "share"
        if unit == "口" or unit.startswith("contract"):
            return "contract"
        if unit in {"枚", "幣"}:
            return "asset"
        # 有些券商把單位只寫在欄名（張數/股數/口數），數值後不再重複。
        if re.search(r"張\s*數|\bLOTS?\b", evidence, re.I):
            return "lot"
        if re.search(r"股\s*數|\bSHARES?\b", evidence, re.I):
            return "share"
        if re.search(r"口\s*數|\bCONTRACTS?\b", evidence, re.I):
            return "contract"
        return "quantity"
    if unit_kind == "pnl":
        # 幣別可能寫在數字前（USD +20）或後（+20 USD）。只看緊鄰數字的
        # 文字，避免把同一行商品代號 BTCUSDT 誤認成損益幣別。
        nearby = f"{raw} {tail[:20]}".upper()
        aliases = (
            (r"NT\$|\b(?:TWD|NTD)\b", "TWD"),
            (r"US\$|\bUSD\b", "USD"),
            (r"\bUSDT\b", "USDT"),
            (r"\bUSDC\b", "USDC"),
            (r"\bBUSD\b", "BUSD"),
            (r"\b(?:RMB|CNY|CNH)\b", "CNY"),
            (r"\bJPY\b", "JPY"),
            (r"\bHKD\b", "HKD"),
            (r"\bEUR\b", "EUR"),
            (r"\bGBP\b", "GBP"),
            (r"\bCHF\b", "CHF"),
            (r"\bCAD\b", "CAD"),
            (r"\bAUD\b", "AUD"),
            (r"\bNZD\b", "NZD"),
            (r"\bSGD\b", "SGD"),
            (r"\bKRW\b", "KRW"),
        )
        for pattern, currency in aliases:
            if re.search(pattern, nearby):
                return currency
        return "currency"
    if unit_kind == "risk_level":
        return "price"
    return unit_kind


def _valid_number_for_rule(value: float, unit: str, rule: _NumberRule) -> bool:
    if rule.field in {"entry_price", "quantity"}:
        return value > 0 and unit != "percent"
    if rule.field in {"exit_price", "buy_price", "sell_price", "fill_price"}:
        # 選擇權到期歸零時，出場/平倉價可能確實是 0；負價格仍拒絕。
        return value >= 0 and unit != "percent"
    if rule.field in {"stop_loss", "take_profit"}:
        # 停損常寫成 -5%，停利常寫 +10%；百分比保留原始正負語意。
        return value != 0 if unit == "percent" else value > 0
    return True


def _candidate_from_number(
    rule: _NumberRule,
    raw: str,
    evidence: str,
    line_number: int,
    tail: str,
) -> FieldCandidate | None:
    value, has_sign, _ = _parse_number(raw)
    if value is None:
        return None
    unit = _number_unit(raw, tail, rule.unit_kind, evidence)
    if not _valid_number_for_rule(value, unit, rule):
        return None

    confidence = rule.confidence
    basis = rule.basis
    if rule.sign_hint is not None:
        if has_sign:
            # Profit / Loss 常只是券商的固定欄名，數值本身仍可能帶相反
            # 正負號。不能讓欄名字義蓋掉明示符號；兩者矛盾時保留原值供
            # 人工核對，但降到安全門檻以下，避免翻轉交易結果。
            sign_conflict = value != 0 and (
                (rule.sign_hint > 0 and value < 0)
                or (rule.sign_hint < 0 and value > 0)
            )
            if sign_conflict:
                confidence = min(confidence, 0.55)
                basis += "；欄名與畫面明示正負號矛盾，必須人工確認"
        else:
            value = abs(value) * rule.sign_hint
    elif rule.unit_kind == "pnl" and value != 0 and not has_sign:
        # 券商常用紅/綠色代表盈虧；OCR 純文字會遺失顏色。沒有 + / - 就不能猜。
        confidence = min(confidence, 0.64)
        basis += "；OCR 未保留正負號（畫面顏色不能當作文字證據），必須人工確認"

    return FieldCandidate(
        field=rule.field,
        candidate_value=value,
        raw_value=raw.strip(),
        confidence=confidence,
        evidence=evidence,
        line_number=line_number,
        basis=basis,
        unit=unit,
    )


def _extract_number_candidates(lines: list[tuple[int, str]]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for rule in _NUMBER_RULES:
        label_re = _label_pattern(rule.labels)
        for index, (_, line) in enumerate(lines):
            # 避免「未實現損益」又被較泛的「損益」規則吃成已實現。
            if rule.field == "pnl" and re.search(
                r"未\s*實\s*現|浮\s*動|未\s*平\s*倉|持\s*倉|帳\s*面|"
                r"\b(?:UNREALIZED|FLOATING|OPEN|POSITION)\b.{0,12}"
                r"\bP\s*(?:N\s*L|&\s*L)\b",
                line,
                re.I,
            ):
                continue
            # 已有明確進/出/買/賣標籤時，不再把同一數字當成 generic 成交價。
            if rule.field == "fill_price" and re.search(
                r"進\s*場|出\s*場|開\s*倉|平\s*倉|買\s*[進入]|賣\s*出", line
            ):
                continue
            for label_match in label_re.finditer(line):
                fragment, evidence, line_no = _value_fragment(lines, index, label_match)
                # 標籤獨占一行時只接受下一行「由數字開始」的值。若下一行是
                # 「出場價: 100」之類另一個欄位，search 會誤把 100 配給前一欄。
                number_match = _NUMBER_TOKEN.match(fragment)
                if number_match is None:
                    continue
                raw = number_match.group("number")
                tail = fragment[number_match.end():]
                candidate = _candidate_from_number(rule, raw, evidence, line_no, tail)
                if candidate is not None:
                    candidates.append(candidate)
    return candidates


def _extract_symbol_candidates(lines: list[tuple[int, str]]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    label_re = _label_pattern(
        ("股票代號", "證券代號", "商品代號", "標的代號", "代號", "SYMBOL", "TICKER", "標的", "商品")
    )
    code_re = re.compile(
        r"(?:\$)?(?:"
        r"\d{4,6}[A-Z]?"
        r"|[A-Z]{1,8}(?:[/_.-]?(?:USDT|USDC|USD|TWD|BTC|ETH))?"
        r"|[A-Z]{1,5}\d{2,8}"
        r")",
        re.I,
    )

    for index, (_, line) in enumerate(lines):
        for match in label_re.finditer(line):
            fragment, evidence, line_no = _value_fragment(lines, index, match)
            code_match = code_re.match(fragment)
            if code_match is None:
                # 支援「台積電 2330」，但仍要求代號緊接商品名稱，避免從
                # 下一個不相干欄位的句中撈到任意數字。
                named_code = re.match(
                    r"[\u4e00-\u9fff]{2,16}\s*[（(]?\s*("
                    + code_re.pattern
                    + r")",
                    fragment,
                    re.I,
                )
                code_match = named_code
            if code_match:
                raw_code = code_match.group(1) if code_match.lastindex else code_match.group(0)
                value = raw_code.lstrip("$").upper()
            else:
                # 明確標籤後的純中文商品名可保留；不擅自查表換成代號。
                name_match = re.match(r"([\u4e00-\u9fff]{2,16})", fragment)
                if name_match is None:
                    continue
                value = name_match.group(1)
            candidates.append(
                FieldCandidate(
                    "symbol",
                    value,
                    fragment,
                    0.98,
                    evidence,
                    line_no,
                    "畫面以標的/商品/代號欄位直接標示",
                )
            )

    # 常見券商標題只有「2330 台積電」。必須同時有代號與名稱才採為候選；
    # 裸數字可能是價格、帳號或日期，因此絕不猜。
    title_patterns = (
        re.compile(r"^(?P<code>\d{4,6}[A-Z]?)\s+[\u4e00-\u9fff]{2,16}$", re.I),
        re.compile(r"^[\u4e00-\u9fff]{2,16}\s+(?P<code>\d{4,6}[A-Z]?)$", re.I),
        re.compile(r"^\$?(?P<code>[A-Z]{1,5})\s+[A-Za-z][A-Za-z .-]{2,24}$", re.I),
    )
    for line_no, line in lines:
        for pattern in title_patterns:
            match = pattern.match(line)
            if not match:
                continue
            code = match.group("code").upper()
            if code.isdigit() and 1900 <= int(code) <= 2100:
                continue
            candidates.append(
                FieldCandidate(
                    "symbol",
                    code,
                    code,
                    0.82,
                    line,
                    line_no,
                    "券商常見的『代號 + 名稱』標題格式；仍應對照原圖",
                )
            )
            break

    crypto_title = re.compile(r"^(?P<code>[A-Z0-9]{2,10}[/_.-]?(?:USDT|USDC|USD|TWD))$", re.I)
    for line_no, line in lines:
        if match := crypto_title.match(line):
            candidates.append(
                FieldCandidate(
                    "symbol",
                    match.group("code").replace("/", "").upper(),
                    match.group("code"),
                    0.84,
                    line,
                    line_no,
                    "畫面標題符合常見加密貨幣交易對格式；仍應對照原圖",
                )
            )
    return candidates


def _extract_direction_candidates(lines: list[tuple[int, str]]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    side_label_re = _label_pattern(
        ("交易方向", "持倉方向", "多空方向", "方向", "多空", "POSITION SIDE", "SIDE")
    )
    action_label_re = _label_pattern(("買賣方向", "買賣別", "交易動作", "成交別", "動作", "ACTION"))

    def side_from(fragment: str) -> tuple[str, str] | None:
        compact = re.sub(r"[：:=\s]", "", fragment).lower()
        long_signal = compact in {"多", "long"} or re.search(
            r"做\s*多|多\s*單|\bLONG\b|\bBUY\s*TO\s*OPEN\b|買\s*進\s*開\s*倉",
            fragment,
            re.I,
        )
        short_signal = compact in {"空", "short"} or re.search(
            r"做\s*空|空\s*單|\bSHORT\b|\bSELL\s*TO\s*OPEN\b|"
            r"賣\s*出\s*開\s*倉|放\s*空",
            fragment,
            re.I,
        )
        if long_signal and short_signal:
            return None
        if long_signal:
            return "long", "畫面明示做多/多單/買進開倉"
        if short_signal:
            return "short", "畫面明示做空/空單/賣出開倉"
        return None

    def action_from(fragment: str) -> tuple[str, str] | None:
        if re.search(r"買\s*進|買\s*入|\bBUY\b", fragment, re.I):
            return "buy", "畫面明示買進動作；買進不一定等於做多，可能是空單回補"
        if re.search(r"賣\s*出|\bSELL\b", fragment, re.I):
            return "sell", "畫面明示賣出動作；賣出不一定等於做空，可能是多單出場"
        return None

    for index, (_, line) in enumerate(lines):
        for match in side_label_re.finditer(line):
            fragment, evidence, line_no = _value_fragment(lines, index, match)
            parsed = side_from(fragment)
            if parsed:
                value, basis = parsed
                candidates.append(FieldCandidate("side", value, fragment, 0.98, evidence, line_no, basis))
            elif action := action_from(fragment):
                # 常見英文券商寫 Side: Buy/Sell。這仍只是成交動作，不把 Buy
                # 偷換成 Long，也不把 Sell 偷換成 Short。
                value, basis = action
                candidates.append(FieldCandidate("action", value, fragment, 0.94, evidence, line_no, basis))
        for match in action_label_re.finditer(line):
            fragment, evidence, line_no = _value_fragment(lines, index, match)
            if parsed := action_from(fragment):
                value, basis = parsed
                candidates.append(FieldCandidate("action", value, fragment, 0.96, evidence, line_no, basis))

        # 沒有欄位標籤時，只接受整行就是足夠明確的方向字樣。不能在
        # 「Ticker: LONG」或商品名稱中看到 LONG/SHORT 就推測持倉方向。
        standalone_direction = re.fullmatch(
            r"(?:多|空|做\s*多|做\s*空|多\s*單|空\s*單|LONG|SHORT|"
            r"BUY\s*TO\s*OPEN|SELL\s*TO\s*OPEN|"
            r"買\s*進\s*開\s*倉|賣\s*出\s*開\s*倉|放\s*空)",
            re.sub(r"[：:=]", "", line).strip(),
            re.I,
        )
        if standalone_direction and (parsed := side_from(standalone_direction.group(0))):
            value, basis = parsed
            candidates.append(FieldCandidate("side", value, value, 0.94, line, lines[index][0], basis))
        stripped = re.sub(r"[：:=]", "", line).strip()
        if re.fullmatch(r"買\s*[進入]|BUY", stripped, re.I):
            candidates.append(
                FieldCandidate(
                    "action", "buy", stripped, 0.90, line, lines[index][0],
                    "獨立一行明示買進；不據此猜測多空方向",
                )
            )
        elif re.fullmatch(r"賣\s*出|SELL", stripped, re.I):
            candidates.append(
                FieldCandidate(
                    "action", "sell", stripped, 0.90, line, lines[index][0],
                    "獨立一行明示賣出；不據此猜測多空方向",
                )
            )
    return candidates


def _normalize_time(raw: str) -> tuple[str, float, str]:
    normalized = _normalize_line(raw)
    has_explicit_roc = normalized.startswith("民國")
    if has_explicit_roc:
        numbers = [int(number) for number in re.findall(r"\d+", normalized)]
        if len(numbers) >= 3:
            year, month, day = numbers[:3]
            time_parts = numbers[3:]
            try:
                dt = datetime(year + 1911, month, day, *(time_parts + [0, 0])[:3])
            except ValueError:
                return normalized, 0.20, "日期格式看似民國年，但數值無效"
            return dt.strftime("%Y-%m-%d %H:%M:%S"), 1.0, "畫面明示民國年，已換算西元"

    chinese = re.fullmatch(
        r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?"
        r"(?:[ T]*(\d{1,2}):(\d{2})(?::(\d{2}))?)?",
        normalized,
    )
    if chinese:
        parts = [int(value) if value else 0 for value in chinese.groups()]
        try:
            dt = datetime(*parts)
        except ValueError:
            return normalized, 0.20, "日期數值無效"
        return dt.strftime("%Y-%m-%d %H:%M:%S"), 1.0, "完整西元日期時間"

    full = re.fullmatch(
        r"(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})"
        r"(?:[ T]*(\d{1,2}):(\d{2})(?::(\d{2}))?)?",
        normalized,
    )
    if full:
        parts = [int(value) if value else 0 for value in full.groups()]
        try:
            dt = datetime(*parts)
        except ValueError:
            return normalized, 0.20, "日期數值無效"
        return dt.strftime("%Y-%m-%d %H:%M:%S"), 1.0, "完整西元日期時間"

    # 115/07/20 可能是民國 115 年，也可能是 OCR 漏掉 20；不得自行換算。
    if re.fullmatch(r"\d{2,3}[/.\-]\d{1,2}[/.\-]\d{1,2}(?:\s+.*)?", normalized):
        return normalized, 0.55, "年份不是四碼且未寫『民國』，不可猜測曆法"
    if re.fullmatch(r"\d{1,2}[/.\-]\d{1,2}(?:\s+.*)?", normalized):
        return normalized, 0.50, "缺少年份，需人工確認"
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", normalized):
        return normalized, 0.45, "只有時間、沒有日期，不能直接建立交易紀錄"
    return normalized, 0.20, "時間格式無法安全正規化"


def _extract_time_candidates(lines: list[tuple[int, str]]) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for field_name, labels, base_confidence, basis in _TIME_RULES:
        label_re = _label_pattern(labels)
        for index, (_, line) in enumerate(lines):
            if field_name == "trade_time" and re.search(r"進\s*場|出\s*場|開\s*倉|平\s*倉", line):
                continue
            for label_match in label_re.finditer(line):
                fragment, evidence, line_no = _value_fragment(lines, index, label_match)
                # 同數字欄位：不跨到下一個有標籤的日期欄位偷取時間。
                time_match = _TIME_TOKEN.match(fragment)
                if time_match is None:
                    continue
                raw = time_match.group("time")
                value, completeness, time_basis = _normalize_time(raw)
                candidates.append(
                    FieldCandidate(
                        field_name,
                        value,
                        raw,
                        min(base_confidence, completeness),
                        evidence,
                        line_no,
                        f"{basis}；{time_basis}",
                        unit="datetime",
                    )
                )
    return candidates


def _extract_strategy_clues(lines: list[tuple[int, str]]) -> tuple[StrategyClue, ...]:
    clues: list[StrategyClue] = []
    seen: set[tuple[str, int]] = set()
    for line_no, line in lines:
        for code, label, pattern in _STRATEGY_PATTERNS:
            if not pattern.search(line) or (code, line_no) in seen:
                continue
            seen.add((code, line_no))
            explicit_reason = bool(re.search(r"策略|進場理由|交易理由|setup|signal", line, re.I))
            clues.append(
                StrategyClue(
                    code=code,
                    label=label,
                    evidence=line,
                    line_number=line_no,
                    confidence=0.92 if explicit_reason else 0.78,
                )
            )
    return tuple(clues)


def _value_key(value: str | float) -> tuple[str, object]:
    if isinstance(value, float):
        return "float", round(value, 10)
    return "str", str(value).strip().lower()


def _deduplicate(candidates: Iterable[FieldCandidate]) -> list[FieldCandidate]:
    best: dict[tuple[str, tuple[str, object]], FieldCandidate] = {}
    for candidate in candidates:
        key = candidate.field, _value_key(candidate.candidate_value)
        previous = best.get(key)
        if previous is None or candidate.confidence > previous.confidence:
            best[key] = candidate
    return list(best.values())


def _derive_entry_exit(candidates: list[FieldCandidate]) -> list[FieldCandidate]:
    """僅在方向明示且買賣價各唯一時，衍生進出場候選。"""

    additions: list[FieldCandidate] = []
    strong_sides = [c for c in candidates if c.field == "side" and c.confidence >= 0.80]
    buys = [c for c in candidates if c.field == "buy_price" and c.confidence >= 0.80]
    sells = [c for c in candidates if c.field == "sell_price" and c.confidence >= 0.80]
    if len({_value_key(c.candidate_value) for c in strong_sides}) != 1:
        return additions
    if len({_value_key(c.candidate_value) for c in buys}) != 1:
        return additions
    if len({_value_key(c.candidate_value) for c in sells}) != 1:
        return additions

    side = strong_sides[0].candidate_value
    buy = max(buys, key=lambda c: c.confidence)
    sell = max(sells, key=lambda c: c.confidence)
    if side == "long":
        entry, exit_ = buy, sell
        mapping = "明示做多，故買進價對應進場、賣出價對應出場"
    elif side == "short":
        entry, exit_ = sell, buy
        mapping = "明示做空，故賣出價對應進場、買進價對應回補出場"
    else:
        return additions

    for field_name, source in (("entry_price", entry), ("exit_price", exit_)):
        additions.append(
            FieldCandidate(
                field_name,
                source.candidate_value,
                source.raw_value,
                min(0.88, source.confidence, strong_sides[0].confidence),
                f"{strong_sides[0].evidence} | {source.evidence}",
                source.line_number,
                mapping,
                unit=source.unit,
                derived=True,
            )
        )
    return additions


def _select_candidates(
    candidates: list[FieldCandidate],
    threshold: float,
) -> tuple[dict[str, FieldCandidate], list[str]]:
    selected: dict[str, FieldCandidate] = {}
    warnings: list[str] = []
    by_field: dict[str, list[FieldCandidate]] = {}
    for candidate in candidates:
        by_field.setdefault(candidate.field, []).append(candidate)

    for field_name, group in by_field.items():
        ordered = sorted(group, key=lambda c: c.confidence, reverse=True)
        top = ordered[0]
        if top.confidence < threshold:
            continue
        if field_name == "quantity" and top.unit == "quantity":
            warnings.append(
                "數量雖可讀取，但畫面未明示股／張／口／幣；"
                "單位可能相差 1000 倍，未自動選取。"
            )
            continue
        conflicts = [
            candidate
            for candidate in ordered[1:]
            if _value_key(candidate.candidate_value) != _value_key(top.candidate_value)
            and candidate.confidence >= max(threshold, top.confidence - 0.05)
        ]
        if conflicts:
            values = ", ".join(str(c.candidate_value) for c in [top, *conflicts[:2]])
            warnings.append(
                f"{FIELD_LABELS.get(field_name, field_name)}出現多個高信心值（{values}），"
                "為避免把不同交易混成一筆，未自動選取。"
            )
            continue
        selected[field_name] = top
    return selected, warnings


def parse_ocr_text(
    text: str,
    *,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
    source: str = "ocr_text",
) -> ScreenshotParseResult:
    """把券商截圖 OCR 文字解析為有信心與證據的欄位候選。

    Args:
        text: OCR 後的純文字；可直接貼上或由任意 OCR 引擎取得。
        acceptance_threshold: 自動帶入欄位的最低信心，預設且最低為 0.80；
                              呼叫端只能提高門檻，不能降低安全底線。
        source: 來源識別字串，供 UI / audit log 顯示。

    信心是「文字抽取可靠度」的保守規則分數，不是圖片真偽機率、詐騙機率，
    也不是策略勝率。
    """

    if not isinstance(text, str):
        raise TypeError("text 必須是字串")
    # 門檻只允許調高、不能調低。否則呼叫端把門檻設成 0.5，就會讓缺年份
    # 的時間或遺失正負號的損益變成「安全值」，等於繞過低信心不猜的承諾。
    if not DEFAULT_ACCEPTANCE_THRESHOLD <= acceptance_threshold <= 1.0:
        raise ValueError(
            f"acceptance_threshold 必須介於 {DEFAULT_ACCEPTANCE_THRESHOLD:.2f} 與 1；"
            "只能調得更保守，不能降低安全門檻"
        )

    lines = _meaningful_lines(text)
    if not lines:
        return ScreenshotParseResult(
            source_text=text,
            candidates=(),
            selected_fields={},
            strategy_clues=(),
            warnings=("OCR 沒有讀到可解析文字，請重新截圖或改用手動填寫。",),
            acceptance_threshold=acceptance_threshold,
            source=source,
        )

    candidates: list[FieldCandidate] = []
    candidates.extend(_extract_symbol_candidates(lines))
    candidates.extend(_extract_direction_candidates(lines))
    candidates.extend(_extract_number_candidates(lines))
    candidates.extend(_extract_time_candidates(lines))
    candidates = _deduplicate(candidates)
    candidates.extend(_derive_entry_exit(candidates))
    candidates = _deduplicate(candidates)

    selected, warnings = _select_candidates(candidates, acceptance_threshold)
    if "unrealized_pnl" in selected and "pnl" not in selected:
        warnings.append("畫面只有未實現/浮動損益；未平倉結果不能當成已實現交易績效。")
    if "fill_price" in selected and not ({"entry_price", "exit_price"} & selected.keys()):
        warnings.append("只辨識到單次成交價，必須確認它是進場還是出場後才能建立交易。")
    if "trade_time" in selected and not ({"entry_time", "exit_time"} & selected.keys()):
        warnings.append("只辨識到單次成交時間，必須確認它是進場還是出場時間。")
    if "action" in selected and "side" not in selected:
        warnings.append("買進/賣出是成交動作，不足以判斷做多或做空；方向未自動猜測。")

    return ScreenshotParseResult(
        source_text=text,
        candidates=tuple(sorted(candidates, key=lambda c: (c.field, -c.confidence, c.line_number or 0))),
        selected_fields=selected,
        strategy_clues=_extract_strategy_clues(lines),
        warnings=tuple(warnings),
        acceptance_threshold=acceptance_threshold,
        source=source,
    )


def ocr_image_text(
    image_path: str | Path,
    *,
    ocr_engine: Callable[[Path], str] | None = None,
    language: str = "chi_tra+eng",
) -> str:
    """從圖片取得 OCR 文字，OCR 套件為可選依賴。

    測試、網站或手機端可注入 ``ocr_engine(path) -> str``。未注入時才嘗試
    ``Pillow`` 與 ``pytesseract``；系統仍需另外安裝 Tesseract 與繁中字庫。
    """

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到截圖檔案: {path}")

    if ocr_engine is not None:
        try:
            text = ocr_engine(path)
        except Exception as exc:  # OCR 引擎錯誤需保留原始原因
            raise OCRUnavailableError(f"自訂 OCR 引擎無法讀取 {path.name}: {exc}") from exc
    else:
        try:
            import pytesseract  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OCRUnavailableError(
                "圖片 OCR 是可選功能。請安裝 Pillow、pytesseract、Tesseract OCR "
                "與繁中字庫，或直接把 OCR 文字傳給 parse_ocr_text。"
            ) from exc
        try:
            with Image.open(path) as image:
                text = pytesseract.image_to_string(image, lang=language)
        except Exception as exc:
            raise OCRUnavailableError(
                f"Tesseract 無法讀取 {path.name}: {exc}。請確認 OCR 主程式與 {language} 字庫。"
            ) from exc

    if not isinstance(text, str):
        raise OCRUnavailableError("OCR 引擎必須回傳字串")
    return text


def parse_screenshot_image(
    image_path: str | Path,
    *,
    ocr_engine: Callable[[Path], str] | None = None,
    language: str = "chi_tra+eng",
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
) -> ScreenshotParseResult:
    """OCR 圖片後執行保守式解析；不會繞過人工確認門檻。"""

    path = Path(image_path)
    text = ocr_image_text(path, ocr_engine=ocr_engine, language=language)
    result = parse_ocr_text(
        text,
        acceptance_threshold=acceptance_threshold,
        source=f"image:{path.name}",
    )
    result.warnings = (
        "OCR 可能讀錯數字、正負號與小數點；儲存前請逐欄對照原始截圖證據。",
        *result.warnings,
    )
    return result


__all__ = [
    "DEFAULT_ACCEPTANCE_THRESHOLD",
    "FIELD_LABELS",
    "PRIMARY_FIELDS",
    "STRATEGY_NOTICE",
    "FieldCandidate",
    "OCRUnavailableError",
    "ScreenshotParseResult",
    "StrategyClue",
    "ocr_image_text",
    "parse_ocr_text",
    "parse_screenshot_image",
]
