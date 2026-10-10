"""把截圖解析結果轉成新手能逐項確認的簡易表單。

本模組不做新的 OCR 推論，只負責把 ``ScreenshotParseResult`` 分成：

* ``auto_filled``：高信心且無衝突，可以先帶入，仍顯示原文證據。
* ``needs_review``：有候選但不夠可靠，表單值保持空白。
* ``missing``：完全沒讀到，改用一句白話問題詢問。

這個區分確保 UI 不會為了「看起來很聰明」而把低信心 OCR 數字偷偷填進
交易紀錄，尤其避免正負號、小數點與買賣方向錯誤翻轉統計結果。
"""

from __future__ import annotations

from dataclasses import dataclass

from .screenshot import FIELD_LABELS, PRIMARY_FIELDS, ScreenshotParseResult


@dataclass(frozen=True)
class BeginnerFormField:
    """新手表單的一格；``value`` 永遠只來自安全選取欄位。"""

    name: str
    label: str
    status: str
    value: str | float | None
    suggested_value: str | float | None
    unit: str | None
    confidence: float | None
    evidence: str | None
    help_text: str


@dataclass(frozen=True)
class ReviewQuestion:
    """UI 可逐題呈現的白話覆核問題。"""

    field: str
    prompt: str
    reason: str
    required: bool
    choices: tuple[str, ...] = ()


_HELP_TEXT = {
    "symbol": "填券商顯示的代號；不要只填暱稱或群組老師說的名稱。",
    "side": "做多是先買後賣；做空是先賣後買。單看『買進』不足以判斷方向。",
    "entry_price": "以實際成交均價為準，不要填委託價或盤中看到的報價。",
    "exit_price": "尚未平倉就留空並標記未平倉，不要用目前市價假裝已實現。",
    "quantity": "連同單位確認：台股可能是股或張，期貨是口，加密貨幣可能有小數。",
    "entry_time": "盡量填完整日期與時間；只有 09:30 無法分辨是哪一天。",
    "exit_time": "以真正平倉成交時間為準，不是下單或委託時間。",
    "pnl": "只填扣完手續費、稅與滑價的已實現淨損益，並確認正負號與帳戶幣別。",
    "stop_loss": "只記錄進場前或進場當下已有的停損；不要事後補一個漂亮數字。",
    "take_profit": "只記錄原先計畫；沒有設定就誠實選『沒有』。",
}

_PROMPTS = {
    "symbol": "這筆交易是哪個標的？請填代號，例如 2330、AAPL 或 BTCUSDT。",
    "side": "你是先買後賣（做多），還是先賣後買（做空）？",
    "entry_price": "實際進場成交均價是多少？",
    "exit_price": "實際出場成交均價是多少？若還沒平倉，請選『尚未平倉』。",
    "quantity": "實際成交數量是多少？請一起確認單位（股／張／口／幣）。",
    "entry_time": "哪一天、幾點進場？請盡量填完整日期與時間。",
    "exit_time": "哪一天、幾點全部平倉？尚未平倉就不要填。",
    "pnl": "券商顯示的『已實現淨損益』是多少？請確認已扣成本及 +／-。",
    "stop_loss": "進場前有設定停損價或停損百分比嗎？沒有就選『沒有』。",
    "take_profit": "進場前有設定停利價或停利百分比嗎？沒有就選『沒有』。",
}


def build_beginner_form(result: ScreenshotParseResult) -> tuple[BeginnerFormField, ...]:
    """將解析結果轉成固定順序的十格表單。

    低信心候選只放在 ``suggested_value``，``value`` 必定保持 ``None``，讓
    前端必須取得使用者明確確認才能存檔。
    """

    fields: list[BeginnerFormField] = []
    for field_name in PRIMARY_FIELDS:
        selected = result.selected_fields.get(field_name)
        candidates = result.candidates_for(field_name)
        top = candidates[0] if candidates else None
        if selected is not None:
            status = "auto_filled"
            value = selected.candidate_value
            suggested = selected.candidate_value
            candidate = selected
        elif top is not None:
            status = "needs_review"
            value = None
            suggested = top.candidate_value
            candidate = top
        else:
            status = "missing"
            value = None
            suggested = None
            candidate = None
        fields.append(
            BeginnerFormField(
                name=field_name,
                label=FIELD_LABELS[field_name],
                status=status,
                value=value,
                suggested_value=suggested,
                unit=candidate.unit if candidate else None,
                confidence=candidate.confidence if candidate else None,
                evidence=candidate.evidence if candidate else None,
                help_text=_HELP_TEXT[field_name],
            )
        )
    return tuple(fields)


def build_review_questions(result: ScreenshotParseResult) -> tuple[ReviewQuestion, ...]:
    """依缺漏與模糊處產生短問題，核心必填先問、紀律欄位後問。"""

    questions: list[ReviewQuestion] = []
    core_required = {"symbol", "side", "entry_price", "exit_price", "quantity"}

    for form_field in build_beginner_form(result):
        if form_field.status == "auto_filled":
            continue
        field_name = form_field.name
        reason = (
            f"OCR 有讀到候選「{form_field.suggested_value}」，但信心或一致性不足，不能自動填入。"
            if form_field.status == "needs_review"
            else "截圖中沒有安全辨識到這個欄位。"
        )
        choices: tuple[str, ...] = ()
        if field_name == "side":
            action = result.value("action")
            if action:
                reason += " 畫面只顯示買進/賣出動作，仍無法知道它是開倉還是平倉。"
            choices = ("做多（先買後賣）", "做空（先賣後買）", "不確定")
        elif field_name == "exit_price":
            choices = ("已平倉，填成交均價", "尚未平倉", "只平倉一部分")
        elif field_name in {"stop_loss", "take_profit"}:
            choices = ("有，填價格", "有，填百分比", "沒有", "不記得")

        questions.append(
            ReviewQuestion(
                field=field_name,
                prompt=_PROMPTS[field_name],
                reason=reason,
                required=field_name in core_required,
                choices=choices,
            )
        )

    # 金額沒有幣別就不能和其他交易相加；即使 OCR 看見 USD/TWD，也必須由
    # 本人確認這是帳戶結算幣別，而不是商品報價幣別或畫面上其他文字。
    pnl_candidates = result.candidates_for("pnl")
    if pnl_candidates:
        detected_unit = pnl_candidates[0].unit
        detected_text = (
            f"OCR 看見可能的幣別「{detected_unit}」，仍需確認是這筆淨損益的帳戶結算幣別。"
            if detected_unit and detected_unit != "currency"
            else "截圖沒有安全辨識到淨損益的帳戶結算幣別。"
        )
        questions.append(
            ReviewQuestion(
                field="pnl_currency",
                prompt="這筆已實現淨損益用哪個幣別結算？例如 TWD、USD 或 USDT。",
                reason=detected_text + " 不同幣別不可直接相加。",
                required=True,
                choices=("TWD", "USD", "USDT", "其他／不確定"),
            )
        )

    # 交易策略不是 OCR 欄位的既定事實：即使掃到技術詞，也要本人確認這是
    # 事前規則，而非圖表介面文字或事後解釋。
    if result.strategy_clues:
        clue_names = "、".join(dict.fromkeys(clue.label for clue in result.strategy_clues))
        prompt = (
            f"畫面出現「{clue_names}」等字樣。這真的是你進場前寫下的規則，"
            "還是圖表上的介面文字？請用一句話確認。"
        )
        reason = "策略線索只做文字描述，不能由 OCR 認證交易技術或統計優勢。"
    else:
        prompt = "你進場前的規則是什麼？請用一句話寫下當時就知道的理由。"
        reason = "沒有事前、可重複的規則，就無法區分策略與事後合理化。"
    questions.append(
        ReviewQuestion(
            field="strategy_reason",
            prompt=prompt,
            reason=reason,
            required=False,
            choices=("自己事前訂的規則", "跟隨他人訊號", "臨場感覺", "不記得"),
        )
    )
    return tuple(questions)


__all__ = [
    "BeginnerFormField",
    "ReviewQuestion",
    "build_beginner_form",
    "build_review_questions",
]
