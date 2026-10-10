"""詐騙話術文字偵測 — 貼上群組對話,掃描話術特徵。

設計原則(這些限制比功能本身更重要):

  1. **不給假百分比。** 只給序數等級(極高/高/中/低/資訊不足)。
     「87.3% 是詐騙」這種數字沒有經過任何標註語料校準,是假精準。

  2. **不用關鍵字密度。** 密度(命中詞 ÷ 總字數)可被「長文夾一個關鍵字」
     稀釋規避,也會讓短促高壓的正常訊息灌到高分。改用
     **「命中了幾種相異的話術類別」** —— 這對長度穩健,也難以操弄。

  3. **雙向否定視窗。** 中文的否定/警示詞常出現在關鍵字**之後**:
     「保證獲利?那都是騙人的」。只看關鍵字前方會把反詐教育文誤判成詐騙。

  4. **討論守門(discussion gate)。** 出現「這是詐騙嗎」「小心」「不要相信」
     這類討論/警示語境時降級,避免把反詐討論本身判成詐騙。

  5. **單一類別上限。** 同一類別重複命中不加成,防止靠複製貼上灌分。

  6. **不做統計顯著性檢定。** 單一則文字沒有抽樣模型,p 值在此無定義。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from .patterns import SCAM_PATTERNS, ScamPattern, find_pattern

# 零寬字元(U+200B/200C/200D、BOM、word joiner):詐騙文案可夾在關鍵字中間規避比對
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")

# 簡體 → 繁體對照(防盲區):詐騙集團常用簡體發文(「保证获利」「老师带单」),
# 而詞庫全為繁體 —— 實測簡體版核心話術原本 0 命中、判「低風險」,完全繞過偵測。
# 只收「簡體專用字」(在繁體中不作為獨立常用字出現的),**刻意排除**
# 干/里/面/后/发 這類簡繁共用但意義不同的字,避免誤改正常繁體文。
# 涵蓋範圍 = 詞庫 + 警示詞 + 守門詞 + 自我否認詞實際用到的異形字。
_S2T = str.maketrans(
    "证获稳赚赔风险师带单报内线飙级进讯学员见图对帐账财冻缴税储砖载汇额时"
    "错过没现场仅谢骗别当话术质举检诈团吗么断识请问这会机着"
    "赢提现认户银转钱续费资审私聊充值个人",
    "證獲穩賺賠風險師帶單報內線飆級進訊學員見圖對帳帳財凍繳稅儲磚載匯額時"
    "錯過沒現場僅謝騙別當話術質舉檢詐團嗎麼斷識請問這會機著"
    "贏提現認戶銀轉錢續費資審私聊充值個人",
)

_SENTENCE_DELIM = re.compile(r"[。!?!?\n；;]+")
_CLAUSE_DELIM = re.compile(r"[,，：:\n]+")

# 限制只有常見的「視覺遮字」可以出現在話術字元之間。不移除一般漢字或數字，
# 避免把原本不相關的文字硬拼成關鍵詞。
_OBFUSCATION_GAP = r"[\s*＊xX×oO○●•·・.．_＿\-—–~～/／\\|｜]"

# ── 話術詞庫:(片語, 對應的詐騙型態代碼, 權重) ────────────────
# 權重 3 = 重大危險訊號用語;2 = 強烈可疑;1 = 需搭配其他訊號
_LEXICON: list[tuple[str, str, int]] = [
    # 保證獲利 / 高勝率話術
    ("保證獲利", "guaranteed_return", 3),
    ("穩賺不賠", "guaranteed_return", 3),
    ("穩賺", "guaranteed_return", 2),
    ("包賺", "guaranteed_return", 3),
    ("零風險", "guaranteed_return", 3),
    ("保本", "guaranteed_return", 2),
    ("穩定獲利", "guaranteed_return", 1),
    ("月配息", "guaranteed_return", 1),
    ("翻倍", "guaranteed_return", 1),
    ("躺著賺", "guaranteed_return", 2),
    # 假飆股群 / 假老師
    ("老師帶單", "fake_stock_group", 3),
    ("帶單", "fake_stock_group", 2),
    ("報明牌", "fake_stock_group", 2),
    ("明牌", "fake_stock_group", 1),
    ("內線", "fake_stock_group", 3),
    ("飆股", "fake_stock_group", 1),
    ("升級vip", "fake_stock_group", 3),
    ("升級 vip", "fake_stock_group", 3),
    ("進二群", "fake_stock_group", 3),
    ("二群", "fake_stock_group", 2),
    ("加客服", "fake_stock_group", 2),
    ("私訊我", "fake_stock_group", 1),
    ("跟單", "fake_stock_group", 1),
    ("代操", "fake_stock_group", 2),
    ("專員帶單", "fake_stock_group", 2),
    # 假名師 / 假績效
    ("學員見證", "fake_guru", 2),
    ("獲利截圖", "fake_guru", 2),
    ("對帳單", "fake_guru", 1),
    ("財富自由", "fake_guru", 1),
    ("素人翻身", "fake_guru", 2),
    ("跟著老師", "fake_guru", 2),
    # 詐騙幣 / 假平台
    ("解凍金", "fake_platform", 3),
    ("認證金", "fake_platform", 3),
    ("驗證金", "fake_platform", 3),
    ("風控金", "fake_platform", 3),
    ("刷流水", "fake_platform", 3),
    ("出金要繳稅", "fake_platform", 3),
    ("繳稅才能出金", "fake_platform", 3),
    ("平台儲值", "fake_platform", 3),
    ("入金", "fake_platform", 2),
    ("充值", "fake_platform", 2),
    ("搬磚套利", "fake_platform", 3),
    ("下載app", "fake_platform", 1),
    ("匯款到", "fake_platform", 2),
    ("轉帳到", "fake_platform", 2),
    ("小額出金", "fake_platform", 2),
    ("無法出金", "withdrawal_block", 2),
    ("出金失敗", "withdrawal_block", 2),
    ("提現失敗", "withdrawal_block", 2),
    ("帳戶凍結", "withdrawal_block", 2),
    ("解除風控", "withdrawal_block", 2),
    ("指定錢包", "money_destination", 2),
    ("usdt", "crypto_payment", 1),
]

# 結構性話術訊號:急迫、稀缺、權威、社會證明
_STRUCTURAL: list[tuple[str, str, int]] = [
    ("名額有限", "urgency", 2),
    ("限時", "urgency", 1),
    ("最後機會", "urgency", 2),
    ("錯過就沒有", "urgency", 2),
    ("現在不進場", "urgency", 2),
    ("僅限今天", "urgency", 2),
    ("恭喜學員", "social_proof", 2),
    ("感謝老師", "social_proof", 2),
    ("又賺了", "social_proof", 1),
]

# 警示詞只用在「明確回指」或引用判斷，不能因同一長句裡碰巧出現
# 「手法／小心／不要相信」就把後面的匯款指令一起洗白。實際命中的否定
# 由 _is_warning_context 以命中前後的位置關係判斷。
_WARNING_MARKERS = (
    "都是騙人", "是騙人的", "都是假", "是假的", "別信", "不要信", "不要相信",
    "不相信", "不可信", "不要匯款", "別匯款", "勿匯款",
    "質疑", "拆穿", "揭穿", "舉報", "檢舉", "報警", "165",
    "是詐騙", "都是詐騙", "常見的詐騙", "典型的詐騙", "受害",
    "都是常見話術", "都是典型話術", "詐騙話術", "詐騙手法",
)

# ⚠️ 自我否認:詐騙犯自己說「我們不是詐騙」。這**不是**否定,反而是警訊
# (protesting too much)。必須在判定警示語境前先剔除,否則會被規避。
_SELF_DENIAL = (
    "不是詐騙", "並非詐騙", "絕不是詐騙", "不是騙人", "不是老鼠會", "不是吸金",
)

# 討論守門詞:整段文字若明顯是在「討論/教育/求證」詐騙,整體降級
_DISCUSSION_MARKERS = (
    "是詐騙嗎", "是不是詐騙", "怎麼判斷", "如何辨識", "反詐", "防詐",
    "請問這", "有人遇過", "求證", "165", "分享一下經驗", "提醒大家",
    "典型手法", "常見手法", "請小心", "詐騙手法",
)

_REPORTING_MARKERS = (
    "他說", "她說", "對方說", "老師說", "群組說", "客服說", "訊息寫",
    "新聞", "報導", "引用", "引述", "範例", "教材", "收到", "轉傳",
)

_QUESTION_MARKERS = (
    "真的嗎", "可信嗎", "可信度", "正常嗎", "合理嗎", "安全嗎", "該相信嗎",
    "這樣對嗎", "會不會是詐騙", "是不是騙人", "真的假的",
)

# LINE 匯出的主要格式為「時間<TAB>發言者<TAB>內容」。也接受常見的
# 「[10:03] 小明：內容」轉貼格式；不把日期、儲存說明或系統事件當成發言。
_TIME_TOKEN = r"(?:(?:上午|下午|am|pm)\s*)?\d{1,2}:\d{2}(?:\s*(?:am|pm))?"
_LINE_TAB = re.compile(
    rf"^\s*(?P<time>{_TIME_TOKEN})\t(?P<speaker>[^\t]{{1,80}})\t(?P<message>.*)$",
    re.IGNORECASE,
)
_LINE_BRACKET = re.compile(
    rf"^\s*\[(?P<time>{_TIME_TOKEN})\]\s*(?P<speaker>[^:：]{{1,80}})[:：]\s*(?P<message>.*)$",
    re.IGNORECASE,
)
_LINE_DATE = re.compile(r"^\s*\d{4}[./-]\d{1,2}[./-]\d{1,2}(?:\s|\(|（|$)")

_WITHDRAWAL_RE = re.compile(r"出金|提現|提領|提款|出款|領回(?:本金|資金)|取回(?:本金|資金)")
_FEE_RE = re.compile(r"稅金?|保證金|認證金|驗證金|解凍金|風控金|手續費|通道費|違約金|資金驗證")
_PAY_RE = re.compile(r"繳|交|支付|補繳|匯款|轉帳|打款")
_CONDITION_RE = re.compile(
    r"先|才能|才可|才會|必須先|需要先|要求先|補繳|否則|"
    r"完成後.{0,8}才|凍結.{0,20}(?:繳|付|匯|交)|風控.{0,20}(?:繳|付|匯|交)"
)
_SAFE_DEDUCTION_RE = re.compile(
    r"不用(?:另外|先)?(?:繳|匯|付)|不需(?:要)?(?:另外|先)?(?:繳|匯|付)|"
    r"無需(?:另外|先)?(?:繳|匯|付)|不必(?:另外|先)?(?:繳|匯|付)|"
    r"直接.{0,12}扣除|從.{0,12}(?:出金|提領|款項).{0,12}扣除"
)
_PRIVATE_TRANSFER_RE = re.compile(
    r"(?:匯款|轉帳|打款|充值|入金).{0,16}"
    r"(?:私人|個人|老師|助理|客服|專員).{0,10}(?:帳戶|戶頭|錢包|地址)"
)
_CRYPTO_TRANSFER_RE = re.compile(
    r"(?:購買|買|充值|轉).{0,8}(?:usdt|泰達幣).{0,20}(?:指定|客服|老師|助理).{0,8}(?:錢包|地址)"
)
_OWN_ACCOUNT_RE = re.compile(
    r"(?:自己|本人|同名|我的|自有).{0,8}(?:個人帳戶|戶頭|錢包|地址)"
)
# 命中本身前後的直接否定。這些規則刻意要求緊貼命中，避免
# 「不要相信銀行客服 現在請匯款到老師帳戶」中的前半句否定後半句。
_DIRECT_NEGATING_PREFIX_RE = re.compile(
    r"(?:不|沒有|沒|無法|不能|不會|絕不|並不|不是|"
    r"(?:不要|別|勿|切勿|請勿)(?:再)?(?:相信|接受|參加|使用)?)\s*$"
)
_DIRECT_WARNING_AFTER_RE = re.compile(
    r"^(?:.{0,16})?(?:"
    r"(?:都|全都|這些?|那(?:些|種)?|上述|以上)?(?:是|屬於)?(?:常見|典型)?(?:的)?"
    r"(?:騙人(?:的)?|假的?|詐騙|陷阱|詐騙話術|詐騙手法)|"
    r"(?:不要|別|勿|切勿|請勿)(?:再)?相信|不可信)"
)
_EDUCATION_PREFIX_RE = re.compile(
    r"(?:詐騙|騙人|假的?).{0,10}(?:話術|手法|常說|會說|聲稱|宣稱).{0,6}$"
)

_LINE_COMBINE_WINDOW_MINUTES = 10
_LINE_COMBINE_MAX_MESSAGES = 6
_LINE_COMBINE_MAX_CHARS = 600


@dataclass
class _TextUnit:
    """一個可獨立判讀的發言單位。

    LINE 匯出時，text 只放訊息本文，不把時間與名稱混入詞彙。combined
    只用於還原同一發言者連續拆開的文字，不跨發言者。
    """

    text: str
    position: int = 0
    speaker: str | None = None
    timestamp: str | None = None
    day: int = 0
    combined: bool = False

    @property
    def source_label(self) -> str:
        bits = [x for x in (self.timestamp, self.speaker) if x]
        return " ".join(bits)


@dataclass(frozen=True)
class Hit:
    """一次話術命中。"""

    phrase: str
    category: str          # 詐騙型態代碼 或 結構性訊號名
    weight: int
    position: int
    negated: bool          # 附近有否定/警示詞 → 不計分
    sentence: str
    matched_text: str = ""
    speaker: str | None = None
    timestamp: str | None = None
    reason: str = ""


@dataclass
class TextScanResult:
    """文字掃描結果。刻意只給序數等級,不給假百分比。"""

    risk_level: str                     # 極高 / 高 / 中 / 低 / 資訊不足
    headline: str
    distinct_categories: int            # 命中的相異話術類別數(主要判準)
    effective_score: int                # 內部分數(僅供門檻,不對外呈現為百分比)
    hits: list[Hit] = field(default_factory=list)
    negated_hits: list[Hit] = field(default_factory=list)
    is_discussion: bool = False
    matched_patterns: list[ScamPattern] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    annotated_sentences: list[tuple[str, list[str]]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "risk_level": self.risk_level,
            "headline": self.headline,
            "distinct_categories": self.distinct_categories,
            "is_discussion": self.is_discussion,
            "hits": [
                {
                    "phrase": h.phrase,
                    "category": h.category,
                    "weight": h.weight,
                    "matched_text": h.matched_text or h.phrase,
                    "excerpt": h.sentence,
                    "speaker": h.speaker,
                    "timestamp": h.timestamp,
                    "reason": h.reason,
                }
                for h in self.hits
            ],
            "negated_hits": [
                {
                    "phrase": h.phrase,
                    "matched_text": h.matched_text or h.phrase,
                    "excerpt": h.sentence,
                    "speaker": h.speaker,
                    "timestamp": h.timestamp,
                }
                for h in self.negated_hits
            ],
            "matched_patterns": [p.code for p in self.matched_patterns],
            "advice": self.advice,
        }


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """回傳每個句子的 (start, end) 位置區間,用於把命中位置對回正確的句子。

    舊版用「第一個含該片語的句子」,當同一片語出現多次時會對錯句子,
    導致警示語境判定用錯上下文。
    """
    spans: list[tuple[int, int]] = []
    last = 0
    for m in _SENTENCE_DELIM.finditer(text):
        if m.start() > last:
            spans.append((last, m.start()))
        last = m.end()
    if last < len(text):
        spans.append((last, len(text)))
    return spans


def _strip_self_denial(s: str) -> str:
    """把「不是詐騙」這類自我否認先拿掉,免得它被當成警示語境。"""
    out = s
    for d in _SELF_DENIAL:
        # 以等長空白取代，保留後續 match position 的座標。
        out = out.replace(d, " " * len(d))
    return out


def _time_to_minutes(token: str | None) -> int | None:
    if not token:
        return None
    t = token.strip().lower().replace(" ", "")
    is_pm = t.startswith("下午") or t.endswith("pm")
    is_am = t.startswith("上午") or t.endswith("am")
    t = re.sub(r"^(?:上午|下午|am|pm)|(?:am|pm)$", "", t)
    try:
        hour_s, minute_s = t.split(":", 1)
        hour, minute = int(hour_s), int(minute_s)
    except (ValueError, TypeError):
        return None
    if not (0 <= minute < 60 and 0 <= hour <= 23):
        return None
    if is_pm and hour < 12:
        hour += 12
    elif is_am and hour == 12:
        hour = 0
    return hour * 60 + minute


def _line_units(text: str) -> tuple[list[_TextUnit], bool]:
    """解析 LINE 匯出，並保留發言者/時間供證據回溯。"""
    units: list[_TextUnit] = []
    day = 0
    offset = 0
    current: _TextUnit | None = None
    detected = False
    time_prefix = re.compile(rf"^\s*{_TIME_TOKEN}\t", re.IGNORECASE)

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if _LINE_DATE.match(stripped):
            day += 1
            current = None
            offset += len(raw_line)
            continue

        m = _LINE_TAB.match(line) or _LINE_BRACKET.match(line)
        if m:
            detected = True
            message = m.group("message").strip()
            current = _TextUnit(
                text=message,
                position=offset,
                speaker=m.group("speaker").strip(),
                timestamp=m.group("time").strip(),
                day=day,
            )
            if message:
                units.append(current)
            offset += len(raw_line)
            continue

        stripped_lower = stripped.lower()
        is_header = (
            stripped_lower.startswith("[line]")
            or stripped_lower.startswith("line ")
            or stripped.startswith("儲存日期")
            or stripped_lower.startswith("saved on")
        )
        # LINE 訊息內的手動換行沒有時間欄，接回前一則。有時間但沒有
        # 「發言者<TAB>內容」的是系統事件，不接進上一則。
        if detected and current is not None and stripped and not is_header and not time_prefix.match(line):
            current.text = f"{current.text}\n{stripped}"
        elif time_prefix.match(line):
            current = None
        offset += len(raw_line)

    if not detected:
        return [], False

    # 組合同一發言者、同一日、十分鐘內的訊息。中間可以有別人插話，
    # 但絕不把別人的文字拼進來；詐騙客服常會在回答受害者問題後才補上
    # 「先繳稅才能出金」。訊息數與字數都有上限，避免把整天對話硬湊成證據。
    combined: list[_TextUnit] = []
    for i, first in enumerate(units):
        first_minute = _time_to_minutes(first.timestamp)
        if first_minute is None:
            continue
        texts = [first.text]
        total_chars = len(first.text)
        for nxt in units[i + 1:]:
            if nxt.day != first.day:
                break
            nxt_minute = _time_to_minutes(nxt.timestamp)
            if nxt_minute is None:
                continue
            elapsed = nxt_minute - first_minute
            if elapsed < 0:
                continue
            if elapsed > _LINE_COMBINE_WINDOW_MINUTES:
                break
            if nxt.speaker != first.speaker:
                continue
            if len(texts) >= _LINE_COMBINE_MAX_MESSAGES:
                break
            if total_chars + 1 + len(nxt.text) > _LINE_COMBINE_MAX_CHARS:
                break
            texts.append(nxt.text)
            total_chars += 1 + len(nxt.text)
            combined.append(_TextUnit(
                text="\n".join(texts),
                position=first.position,
                speaker=first.speaker,
                timestamp=f"{first.timestamp}-{nxt.timestamp}",
                day=first.day,
                combined=True,
            ))
    return units + combined, True


def _text_units(text: str) -> tuple[list[_TextUnit], bool]:
    units, is_line = _line_units(text)
    if is_line:
        return units, True
    spans = _sentence_spans(text)
    return [
        _TextUnit(text=text[a:b].strip(), position=a)
        for a, b in spans if text[a:b].strip()
    ], False


@lru_cache(maxsize=256)
def _term_pattern(phrase: str) -> re.Pattern[str]:
    """容許全半形正規化後的空白/星號/X/點號遮字，但不略過一般文字。"""
    compact = re.sub(rf"{_OBFUSCATION_GAP}+", "", phrase.lower())
    gap = rf"(?:{_OBFUSCATION_GAP}){{0,3}}"
    return re.compile(gap.join(re.escape(ch) for ch in compact), re.IGNORECASE)


def _clause_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    last = 0
    for m in _CLAUSE_DELIM.finditer(text):
        if m.start() > last:
            spans.append((last, m.start()))
        last = m.end()
    if last < len(text):
        spans.append((last, len(text)))
    return spans or [(0, len(text))]


def _inside_quote(text: str, start: int, end: int) -> bool:
    for left, right in (("「", "」"), ("『", "』"), ("“", "”"), ("'", "'"), ('"', '"')):
        left_pos = text.rfind(left, 0, start + 1)
        if left_pos >= 0 and text.find(right, end) >= 0:
            return True
    line_start = text.rfind("\n", 0, start) + 1
    return text[line_start:start].lstrip().startswith(">")


def _has_warning(text: str) -> bool:
    # 「小心錯過」是急迫銷售話術，不是風險警示。
    cleaned = text.replace("小心錯過", "").replace("小心錯失", "")
    return any(w in cleaned for w in _WARNING_MARKERS)


def _is_warning_context(text_lower: str, start: int, end: int) -> bool:
    """判斷命中處是否在否定、求證或反詐引用語境。

    否定只套用在命中所在子句，或明確回指前句的「這些都是詐騙」。
    因此「不要相信銀行，請匯款到私人帳戶」的後半句仍會被抓到。
    """
    cleaned = _strip_self_denial(text_lower)
    spans = _clause_spans(cleaned)
    idx = next((i for i, (a, b) in enumerate(spans) if a <= start < b), 0)
    a, b = spans[idx]
    prefix = cleaned[max(a, start - 28):start]
    suffix = cleaned[end:min(b, end + 40)]
    if _DIRECT_NEGATING_PREFIX_RE.search(prefix):
        return True
    if _EDUCATION_PREFIX_RE.search(prefix):
        return True
    if _DIRECT_WARNING_AFTER_RE.search(suffix):
        return True

    if idx > 0:
        previous = cleaned[spans[idx - 1][0]:spans[idx - 1][1]].strip()
        # 只接受無受詞的短指令；「不要相信銀行客服」不會影響下一句。
        if re.fullmatch(r"(?:請|千萬)?(?:不要|別|切勿)?(?:相信|匯款|轉帳)?(?:小心|當心|警惕)?", previous) and _has_warning(previous):
            return True
    if idx + 1 < len(spans):
        following = cleaned[spans[idx + 1][0]:spans[idx + 1][1]].strip()
        if re.match(r"(?:這|這些|這種|那|那些|以上|上述|全都|都是)", following) and _has_warning(following):
            return True

    # 只有「明確引用 + 報導/轉述 + 質疑」才降級；騙徒用引號強調承諾仍會命中。
    if _inside_quote(text_lower, start, end):
        has_reporting = any(m in text_lower for m in _REPORTING_MARKERS)
        has_question = any(m in text_lower for m in _QUESTION_MARKERS)
        has_education = any(m in cleaned for m in _DISCUSSION_MARKERS) or _has_warning(cleaned)
        if has_reporting and (has_question or has_education):
            return True
    return False


_CATEGORY_REASON = {
    "guaranteed_return": "使用保證、穩賺或零風險的獲利承諾",
    "fake_stock_group": "以帶單、明牌、VIP 或客服導流",
    "fake_guru": "以學員見證或績效截圖建立信任",
    "fake_platform": "引導入金、匯款或不尋常的平台操作",
    "withdrawal_block": "提及無法出金、凍結或解除風控",
    "withdrawal_precondition": "把另外付費當成出金或提領的前置條件",
    "money_destination": "要求把資金轉到私人、個人或指定錢包",
    "crypto_payment": "引導使用 USDT 等虛擬資產轉帳",
    "urgency": "使用限時、稀缺或最後機會施壓",
    "social_proof": "以群眾感謝或學員獲利營造跟從壓力",
    "self_denial": "主動強調「不是詐騙」，不能作為安全證明",
}


def _hit_for_match(
    phrase: str,
    category: str,
    weight: int,
    unit: _TextUnit,
    start: int,
    end: int,
    matched_text: str,
) -> Hit:
    negated = _is_warning_context(unit.text.lower(), start, end)
    excerpt = " ".join(unit.text.split())[:160]
    return Hit(
        phrase=phrase,
        category=category,
        weight=weight,
        position=unit.position + start,
        negated=negated,
        sentence=excerpt,
        matched_text=matched_text,
        speaker=unit.speaker,
        timestamp=unit.timestamp,
        reason=_CATEGORY_REASON.get(category, "命中需連同其他語境判讀的話術特徵"),
    )


def _span_excerpt(unit: _TextUnit, matches: list[re.Match[str]]) -> tuple[int, int, str]:
    start = min(m.start() for m in matches)
    end = max(m.end() for m in matches)
    lo, hi = max(0, start - 18), min(len(unit.text), end + 18)
    return start, end, " ".join(unit.text[lo:hi].split())


def _composite_hits(units: list[_TextUnit]) -> list[Hit]:
    """假平台常把危險指令拆成數句；這些組合規則只在同一 TextUnit 內運作。"""
    found: dict[str, Hit] = {}
    for unit in units:
        lower = unit.text.lower()
        withdrawal = _WITHDRAWAL_RE.search(lower)
        fee = _FEE_RE.search(lower)
        pay = _PAY_RE.search(lower)
        condition = _CONDITION_RE.search(lower)
        safe_deduction = _SAFE_DEDUCTION_RE.search(lower)
        contrary_after_safe = re.search(
            r"(?:但|但是|不過|然而).{0,30}(?:先|必須|需要|補繳).{0,20}"
            r"(?:稅|保證金|認證金|驗證金|手續費)",
            lower,
        )
        if withdrawal and fee and pay and condition and (not safe_deduction or contrary_after_safe):
            start, end, excerpt = _span_excerpt(unit, [withdrawal, fee, pay, condition])
            h = _hit_for_match(
                "(出金前另行付費)", "withdrawal_precondition", 3,
                unit, start, end, excerpt,
            )
            if h.category not in found or (found[h.category].negated and not h.negated):
                found[h.category] = h

        for regex in (_PRIVATE_TRANSFER_RE, _CRYPTO_TRANSFER_RE):
            m = regex.search(lower)
            if not m:
                continue
            # 「自己的個人帳戶」只有出現在這次轉帳的目的地片段內才是
            # 合法自轉例外。若它只是來源（例如「從自己的帳戶匯到老師帳戶」），
            # 絕不能洗掉真正的私人收款目的地。
            if _OWN_ACCOUNT_RE.search(m.group(0)):
                continue
            category = "money_destination"
            h = _hit_for_match(
                "(轉入私人或指定收款處)", category, 3,
                unit, m.start(), m.end(), " ".join(unit.text[m.start():m.end()].split()),
            )
            if category not in found or (found[category].negated and not h.negated):
                found[category] = h
    return list(found.values())
def scan_text(text: str) -> TextScanResult:
    """掃描一段文字的詐騙話術特徵。

    Returns:
        TextScanResult(序數風險等級 + 逐句標註 + 對應的反詐金句)
    """
    raw = (text or "").strip()
    if len(raw) < 4:
        return TextScanResult(
            risk_level="資訊不足",
            headline="文字太短,無法判斷。請貼上完整的對話或文案。",
        )

    # ── 正規化(防規避):純標準庫,零依賴 ──
    # (a) NFKC:全形拉丁字母/數字/標點 → 半形(「ＶＩＰ」→「vip」、「保　證」的全形空白)
    # (b) 移除零寬字元:詐騙文案可在關鍵字中間夾 U+200B 等,肉眼看不出、比對卻失效
    norm = unicodedata.normalize("NFKC", raw)
    norm = _ZERO_WIDTH.sub("", norm)
    # 簡→繁正規化:str.translate 是 1:1 等長替換,不位移證據座標;
    # 繁體輸入經過此步完全不變。
    norm = norm.translate(_S2T)
    lower = norm.lower()
    units, _ = _text_units(norm)

    # 「是否為討論」只作說明資訊，不再整段一律降級。否定/引用必須和
    # 每個命中處於同一發言單位，避免「請問這是詐騙嗎」被濫用成降級密碼。
    lower_clean = _strip_self_denial(lower)
    is_discussion = any(m in lower_clean for m in _DISCUSSION_MARKERS)

    hits: list[Hit] = []
    negated: list[Hit] = []
    all_terms = (
        [(p, c, w) for p, c, w in _LEXICON]
        + [(p, c, w) for p, c, w in _STRUCTURAL]
        + [(p, "self_denial", 2) for p in _SELF_DENIAL]
    )

    for phrase, category, weight in all_terms:
        # 掃描「所有」出現位置,不能只看第一個就 break:
        # 攻擊情境:「保證獲利?都是騙人的…不,我們真的保證獲利!」——
        # 第一次出現在警示語境會被排除,若就此 break,第二次真正的話術會漏抓。
        # 只要有任何一個 occurrence 不在警示語境,就算一次有效命中(仍只計一次分)。
        best_hit: Hit | None = None
        pattern = _term_pattern(phrase)
        for unit in units:
            for m in pattern.finditer(unit.text.lower()):
                h = _hit_for_match(
                    phrase, category, weight, unit, m.start(), m.end(),
                    " ".join(unit.text[m.start():m.end()].split()),
                )
                if not h.negated:
                    best_hit = h
                    break        # 找到有效命中即可(同片語只計一次分)
                if best_hit is None:
                    best_hit = h # 暫存被否定的,若全部被否定就歸 negated
            if best_hit is not None and not best_hit.negated:
                break
        if best_hit is not None:
            (negated if best_hit.negated else hits).append(best_hit)

    # 跨詞/跨訊息的組合證據。被否定的組合仍保留在 negated_hits，但不計分。
    for h in _composite_hits(units):
        (negated if h.negated else hits).append(h)

    # ── 計分:單一類別只取該類別的最高權重(上限),避免同類堆疊灌分 ──
    per_category_max: dict[str, int] = {}
    for h in hits:
        per_category_max[h.category] = max(per_category_max.get(h.category, 0), h.weight)

    effective = sum(per_category_max.values())
    distinct = len(per_category_max)
    has_hard_evidence = any(w >= 3 for w in per_category_max.values())

    # ── 序數等級:只呈現極高/高/中/低，不把內部權重轉成詐騙百分比 ──
    if has_hard_evidence and distinct >= 2:
        level = "極高"
        headline = "🚨 極高風險:出現多種高危投資話術，且含重大危險訊號。"
    elif has_hard_evidence or distinct >= 3:
        level = "高"
        headline = "⚠️ 高風險:出現需要立即查證的典型高危話術。"
    elif distinct >= 2 or effective >= 3:
        level = "中"
        headline = "🟡 中度可疑:出現一些常見的詐騙話術特徵,請提高警覺。"
    elif distinct >= 1:
        level = "低"
        headline = "🟢 低度警訊:僅出現少量可疑用語，不代表安全。"
    elif negated or is_discussion:
        level = "低"
        headline = "🟢 這段文字主要是求證、引用或警示語境，相關字句未納入風險計分。"
    else:
        level = "低"
        headline = "🟢 未偵測到明顯的詐騙話術。但話術會演化,仍請保持警覺。"

    # ── 對應的反詐金句 ──
    category_pattern = {
        "withdrawal_block": "fake_platform",
        "withdrawal_precondition": "fake_platform",
        "money_destination": "fake_platform",
        "crypto_payment": "fake_platform",
    }
    scam_codes = {category_pattern.get(h.category, h.category) for h in hits}
    matched = [p for p in SCAM_PATTERNS if p.code in scam_codes]

    advice: list[str] = []
    if level in ("極高", "高"):
        advice.append("不要匯款、不要下載對方指定的 App、不要『升級』。")
        advice.append("保留對話與匯款紀錄。台灣可撥打 165 反詐騙諮詢專線查證。")
    if any(h.category == "withdrawal_precondition" for h in hits):
        advice.append("正常稅費通常不會要求先匯至私人帳戶才能出金；請先停止付款並向 165 查證。")
    for p in matched:
        advice.append(f"【相似話術類型:{p.name}】{p.rebuttal}")
    if negated:
        advice.append(
            f"註:有 {len(negated)} 處關鍵字位於否定、警示或明確引用語境，"
            "已保留供人工核對，但未納入風險計分。"
        )

    # ── 可回溯證據:LINE 格式保留時間/發言者，遮字命中保留實際文字 ──
    annotated_map: dict[str, list[str]] = {}
    for h in hits:
        source = " ".join(x for x in (h.timestamp, h.speaker) if x)
        label = f"[{source}] {h.sentence}" if source else h.sentence
        evidence = h.matched_text or h.phrase
        if evidence not in annotated_map.setdefault(label, []):
            annotated_map[label].append(evidence)
    annotated = list(annotated_map.items())

    return TextScanResult(
        risk_level=level,
        headline=headline,
        distinct_categories=distinct,
        effective_score=effective,
        hits=hits,
        negated_hits=negated,
        is_discussion=is_discussion,
        matched_patterns=matched,
        advice=advice,
        annotated_sentences=annotated,
    )


def render_scan(r: TextScanResult) -> str:
    """輸出可讀報告。"""
    L = ["=" * 60, "        詐騙話術文字偵測", "=" * 60, ""]
    L.append(f"【風險等級】{r.risk_level}")
    L.append(f"  {r.headline}")
    L.append("")
    if r.hits:
        L.append("【可核對的對話證據】")
        for h in r.hits[:10]:
            source = " ".join(x for x in (h.timestamp, h.speaker) if x)
            prefix = f"[{source}] " if source else ""
            L.append(f"  「{prefix}{h.sentence[:80]}」")
            L.append(f"      ↳ 命中:{h.matched_text or h.phrase}；理由:{h.reason}")
        L.append("")
    if r.advice:
        L.append("【反詐提醒】")
        for a in r.advice:
            L.append(f"  • {a}")
        L.append("")
    L.append("─" * 60)
    L.append("註:本工具只給序數風險等級，不產生「詐騙機率」或直接定罪。")
    L.append("    低等級不代表安全；話術會演化，最終請以 165 專線查證為準。")
    return "\n".join(L)
