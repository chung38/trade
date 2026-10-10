"""互動式詐騙風險自我檢測清單。

讓使用者回答一連串「是 / 否」問題,依命中的高風險特徵,給出詐騙風險評分
與對應的反詐提醒。可由 CLI 互動執行,也可程式化傳入答案。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .patterns import SCAM_PATTERNS, ScamPattern, find_pattern


@dataclass
class CheckItem:
    """一道檢測題。"""

    key: str
    question: str
    pattern_code: str        # 對應的詐騙型態
    weight: int              # 命中時的風險分數(高風險特徵權重高)


# 檢測題庫:每題對應一種詐騙特徵。weight 越高 = 越重大的危險訊號。
CHECK_ITEMS: list[CheckItem] = [
    CheckItem("pulled_in", "是否有人『主動』把你拉進投資 LINE / Telegram 群?",
              "fake_stock_group", 2),
    CheckItem("teacher_calls", "群裡是否有『老師 / 分析師』每天報明牌、帶你進出場?",
              "fake_stock_group", 2),
    CheckItem("upgrade_vip", "是否被要求『升級 VIP / 進二群 / 付費才能拿到更準的單』?",
              "fake_stock_group", 3),
    CheckItem("member_screenshots", "群裡是否充斥『學員獲利截圖、感謝老師』的訊息?",
              "fake_guru", 2),
    CheckItem("guaranteed", "對方是否宣稱『保證獲利 / 穩賺不賠 / 零風險高報酬』?",
              "guaranteed_return", 3),
    CheckItem("high_winrate_only", "對方是否只強調『高勝率』,卻從不談『賠的時候賠多少』?",
              "guaranteed_return", 2),
    CheckItem("urgency", "是否被催促『現在不進場就錯過』,讓你來不及冷靜思考?",
              "guaranteed_return", 1),
    CheckItem("guru_flex", "對方是否炫耀豪車 / 名錶 / 財富自由,暗示跟著就能致富?",
              "fake_guru", 1),
    CheckItem("only_screenshots", "你看到的『績效』是否只有精選截圖,沒有完整連續的紀錄?",
              "fake_guru", 2),
    CheckItem("transfer_platform", "是否被要求把錢匯到某個『平台 / 錢包 / 客服指定帳戶』?",
              "fake_platform", 3),
    CheckItem("cant_withdraw", "是否出金時被要求『先繳稅 / 刷流水 / 付解凍金』才能領回?",
              "fake_platform", 3),
    CheckItem("small_withdraw_bait", "是否一開始能小額出金,等你加碼後就領不出來?",
              "fake_platform", 3),
]


@dataclass
class ScamCheckResult:
    """檢測結果。"""

    score: int                              # 命中總風險分數
    max_score: int                          # 滿分
    risk_level: str                         # "極高" | "高" | "中" | "低"
    hit_patterns: list[ScamPattern] = field(default_factory=list)
    headline: str = ""
    advice: list[str] = field(default_factory=list)

    @property
    def weighted_ratio(self) -> float:
        """僅供內部門檻判斷用,**不可對使用者呈現為百分比**。

        這個比例是「命中權重 / 總權重」,權重是我們自己訂的,
        沒有經過任何標註語料的校準。把它印成「你有 48% 的機率遇到詐騙」
        是假精準 —— 那個數字不對應任何真實機率,正是本工具譴責的偽科學。
        對外一律只呈現序數等級(極高 / 高 / 中 / 低)。
        """
        return self.score / self.max_score if self.max_score else 0.0


def evaluate(answers: dict[str, bool]) -> ScamCheckResult:
    """依答案(key -> 是否命中)計算詐騙風險。"""
    max_score = sum(it.weight for it in CHECK_ITEMS)
    score = 0
    hit_codes: set[str] = set()
    # 任何一題權重 3(重大危險訊號)被命中,直接視為極高風險
    hard_hit = False
    for it in CHECK_ITEMS:
        if answers.get(it.key):
            score += it.weight
            hit_codes.add(it.pattern_code)
            if it.weight >= 3:
                hard_hit = True

    hit_patterns = [p for p in SCAM_PATTERNS if p.code in hit_codes]

    pct = score / max_score if max_score else 0.0
    if hard_hit or pct >= 0.5:
        risk_level = "極高"
        headline = "🚨 詐騙風險極高:你描述的情境命中了典型投資詐騙的關鍵特徵。"
    elif pct >= 0.3:
        risk_level = "高"
        headline = "⚠️ 詐騙風險偏高:出現多個常見詐騙特徵,請高度警覺。"
    elif pct > 0:
        risk_level = "中"
        headline = "🟡 有一些可疑徵兆,建議對照下方反詐提醒再三確認。"
    else:
        risk_level = "低"
        headline = "✅ 目前沒有命中明顯的詐騙特徵 — 但仍請保持基本警覺。"

    advice: list[str] = []
    if score > 0:
        advice.append(
            "把對方給過的所有『推薦 / 帶單』完整記錄下來(含失敗的),"
            "用本工具 analyze 算期望值與顯著性 —— 別只看他挑給你的成功案例。"
        )
    for p in hit_patterns:
        advice.append(f"【{p.name}】{p.rebuttal}")
    if risk_level in ("極高", "高"):
        advice += [
            "不要再匯任何錢、不要『升級』、不要下載對方指定的 App。",
            "保留對話與匯款紀錄。台灣可撥打『165 反詐騙諮詢專線』查證與求助。",
        ]

    return ScamCheckResult(
        score=score,
        max_score=max_score,
        risk_level=risk_level,
        hit_patterns=hit_patterns,
        headline=headline,
        advice=advice,
    )


def run_scam_check(input_fn=input, output_fn=print) -> ScamCheckResult | None:
    """互動式跑一遍檢測清單。回傳結果;輸入中斷時回傳 None(不給半套結論)。

    input_fn / output_fn 可注入,方便測試;預設用標準輸入輸出。
    """
    output_fn("=" * 60)
    output_fn("           反詐投資王 — 投資詐騙風險自我檢測")
    output_fn("=" * 60)
    output_fn("請依你目前遇到的情況,回答以下問題(y = 是 / n = 否):\n")

    yes_tokens = ("y", "yes", "是", "1", "有")
    no_tokens = ("n", "no", "否", "0", "沒有", "無")
    answers: dict[str, bool] = {}
    for i, it in enumerate(CHECK_ITEMS, 1):
        # 輸入驗證迴圈:看不懂的輸入**不可**靜默當成「否」——
        # 按錯鍵就降低風險評估,對反詐工具是最危險的失敗模式。
        while True:
            try:
                raw = str(input_fn(f"  {i}. {it.question} [y/n] ")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                output_fn(
                    "\n⚠️ 輸入中斷:問卷未完成,不給任何風險結論 —— "
                    "半份問卷算出的『低風險』是假保證。請重新執行 scam-check。"
                )
                return None
            if raw in yes_tokens:
                answers[it.key] = True
                break
            if raw in no_tokens:
                answers[it.key] = False
                break
            output_fn("     ↳ 請輸入 y(是)或 n(否)。看不懂的輸入不會被當成『否』。")

    result = evaluate(answers)

    output_fn("")
    output_fn("=" * 60)
    # 只呈現「命中幾項 / 共幾項」的事實與序數等級。
    # 不印百分比 —— 那個數字沒有經過校準,不對應任何真實機率(假精準)。
    n_hit = sum(1 for it in CHECK_ITEMS if answers.get(it.key))
    n_hard = sum(
        1 for it in CHECK_ITEMS if answers.get(it.key) and it.weight >= 3
    )
    output_fn(f"  命中 {n_hit} / {len(CHECK_ITEMS)} 項詐騙特徵"
              f"(其中重大危險訊號 {n_hard} 項)  風險等級:{result.risk_level}")
    # 可解釋性:等級是「特徵嚴重度加權」的結果,不是純命中數 ——
    # 否則使用者會困惑「同樣中 3 項,為什麼他是中、我是極高」。
    output_fn("  (風險等級依特徵嚴重度加權判定,非單純命中數;"
              "任一重大危險訊號命中即列為極高。)")
    output_fn(f"  {result.headline}")
    output_fn("=" * 60)
    if result.advice:
        output_fn("\n【給你的反詐提醒】")
        for a in result.advice:
            output_fn(f"  • {a}")
    output_fn(
        "\n記住:真正的投資優勢經得起統計檢定與時間考驗,"
        "不需要靠『拉群、保證、急迫感』來說服你。"
    )
    return result
