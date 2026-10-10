"""券商截圖 OCR 與新手覆核表單的安全性測試。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingest.beginner import build_beginner_form, build_review_questions
from core.ingest.screenshot import STRATEGY_NOTICE, parse_ocr_text, parse_screenshot_image


def _field(form, name):
    return next(item for item in form if item.name == name)


def test_explicit_labels_extract_complete_trade_with_evidence():
    text = """
    股票代號：2330 台積電
    方向：做多
    進場價：980.5
    出場價：1,020
    成交數量：1,000 股
    進場時間：2026/07/01 09:05:00
    出場時間：2026/07/03 13:20:00
    已實現損益：+39,500
    停損：960
    停利：1,040
    進場理由：突破前高、站上 20 日均線，RSI 55
    """

    result = parse_ocr_text(text)

    assert result.value("symbol") == "2330"
    assert result.value("side") == "long"
    assert result.value("entry_price") == 980.5
    assert result.value("exit_price") == 1020.0
    assert result.value("quantity") == 1000.0
    assert result.selected_fields["quantity"].unit == "share"
    assert result.value("entry_time") == "2026-07-01 09:05:00"
    assert result.value("exit_time") == "2026-07-03 13:20:00"
    assert result.value("pnl") == 39500.0
    assert result.value("stop_loss") == 960.0
    assert result.value("take_profit") == 1040.0
    assert "進場價" in result.selected_fields["entry_price"].evidence
    assert result.needs_manual_review
    assert result.requires_human_review
    assert result.missing_required_fields == ()

    clue_codes = {clue.code for clue in result.strategy_clues}
    assert {"breakout", "moving_average", "rsi"}.issubset(clue_codes)
    assert "不代表" in STRATEGY_NOTICE
    assert "統計優勢" in STRATEGY_NOTICE


def test_unsigned_pnl_is_preserved_but_never_auto_filled():
    result = parse_ocr_text("股票代號: 2330\n已實現損益: 12,300")

    assert result.value("pnl") is None
    candidate = result.candidates_for("pnl")[0]
    assert candidate.candidate_value == 12300.0
    assert candidate.confidence < result.acceptance_threshold
    assert "正負號" in candidate.basis

    pnl_field = _field(build_beginner_form(result), "pnl")
    assert pnl_field.status == "needs_review"
    assert pnl_field.value is None
    assert pnl_field.suggested_value == 12300.0
    assert pnl_field.evidence == "已實現損益: 12,300"


def test_safety_threshold_can_only_be_raised():
    try:
        parse_ocr_text("已實現損益: 12,300", acceptance_threshold=0.50)
    except ValueError as exc:
        assert "不能降低安全門檻" in str(exc)
    else:
        raise AssertionError("低門檻不應被接受")


def test_buy_action_and_generic_fill_do_not_guess_side_or_entry():
    result = parse_ocr_text(
        "股票代號: 2330\n買賣別: 買進\n成交價: 1,000\n成交時間: 09:31:05"
    )

    assert result.value("action") == "buy"
    assert result.value("fill_price") == 1000.0
    assert result.value("side") is None
    assert result.value("entry_price") is None
    assert result.value("exit_price") is None
    assert result.value("trade_time") is None  # 只有時刻、沒有日期，不能自動帶入
    assert any("不足以判斷做多或做空" in warning for warning in result.warnings)
    assert any("單次成交價" in warning for warning in result.warnings)

    side_question = next(q for q in build_review_questions(result) if q.field == "side")
    assert "開倉還是平倉" in side_question.reason


def test_explicit_short_maps_unique_sell_and_buy_prices_to_entry_exit():
    result = parse_ocr_text(
        "股票代號: AAPL\n方向: 做空\n賣出均價: 200\n買進均價: 180\n成交數量: 10 股"
    )

    assert result.value("side") == "short"
    assert result.value("entry_price") == 200.0
    assert result.value("exit_price") == 180.0
    assert result.selected_fields["entry_price"].derived
    assert "明示做空" in result.selected_fields["entry_price"].basis


def test_percentage_stop_and_zero_option_exit_preserve_units():
    result = parse_ocr_text("標的代號: TXO\n進場價: 25\n出場價: 0\n停損: -5%\n停利: +10%")

    assert result.value("exit_price") == 0.0
    assert result.value("stop_loss") == -5.0
    assert result.selected_fields["stop_loss"].unit == "percent"
    assert result.value("take_profit") == 10.0
    assert result.selected_fields["take_profit"].unit == "percent"


def test_english_broker_labels_and_label_only_quantity_unit():
    result = parse_ocr_text(
        "Ticker: AAPL\nPosition Side: Long\nEntry Price: 200\nExit Price: 210\n"
        "Shares: 2\nEntry Time: 2026-07-19 09:30\nExit Time: 2026-07-20 13:00\n"
        "Realized PnL: +20\nSL: -5%\nTP: +10%"
    )

    assert result.value("symbol") == "AAPL"
    assert result.value("side") == "long"
    assert result.value("entry_price") == 200.0
    assert result.value("exit_price") == 210.0
    assert result.value("quantity") == 2.0
    assert result.selected_fields["quantity"].unit == "share"
    assert result.value("pnl") == 20.0


def test_quantity_without_unit_is_never_auto_selected():
    result = parse_ocr_text(
        "標的代號: 2330\n方向: 做多\n進場價: 980\n出場價: 1000\n數量: 1"
    )

    assert result.value("quantity") is None
    assert "quantity" in result.missing_required_fields
    assert any("單位可能相差 1000 倍" in warning for warning in result.warnings)
    question = next(q for q in build_review_questions(result) if q.field == "quantity")
    assert question.required


def test_macd_cross_is_not_mislabeled_as_moving_average():
    result = parse_ocr_text("策略: MACD 黃金交叉")
    codes = {clue.code for clue in result.strategy_clues}

    assert "macd" in codes and "cross_signal" in codes
    assert "moving_average" not in codes


def test_side_buy_is_action_not_long_even_in_english():
    result = parse_ocr_text("Ticker: AAPL\nSide: Buy\nPrice: 200\nShares: 2")

    assert result.value("action") == "buy"
    assert result.value("side") is None
    assert result.value("entry_price") is None


def test_conflicting_high_confidence_values_are_not_selected():
    result = parse_ocr_text("股票代號: 2330\n股票代號: 2317\n方向: 做多")

    assert result.value("symbol") is None
    assert {c.candidate_value for c in result.candidates_for("symbol")} == {"2330", "2317"}
    assert any("多個高信心值" in warning for warning in result.warnings)


def test_label_on_previous_line_and_spaced_ocr_label_are_supported():
    result = parse_ocr_text("進 場 價\n1,002.5\n停 損\n980")

    assert result.value("entry_price") == 1002.5
    assert result.value("stop_loss") == 980.0
    assert "↵" in result.selected_fields["entry_price"].evidence


def test_empty_label_does_not_steal_value_from_next_labeled_field():
    result = parse_ocr_text("進場價\n出場價: 1,050\n進場時間\n出場時間: 2026/07/20 13:30")

    assert result.value("entry_price") is None
    assert result.value("exit_price") == 1050.0
    assert result.value("entry_time") is None
    assert result.value("exit_time") == "2026-07-20 13:30:00"


def test_ambiguous_roc_year_and_bare_number_are_not_guessed():
    result = parse_ocr_text("2330\n進場時間: 115/07/20 09:30")

    assert result.value("symbol") is None
    assert result.value("entry_time") is None
    assert result.candidates_for("symbol") == ()
    time_candidate = result.candidates_for("entry_time")[0]
    assert time_candidate.candidate_value == "115/07/20 09:30"
    assert time_candidate.confidence < result.acceptance_threshold
    assert "不可猜測曆法" in time_candidate.basis


def test_explicit_roc_year_is_safely_normalized():
    result = parse_ocr_text("進場時間: 民國115年7月20日 09:30")

    assert result.value("entry_time") == "2026-07-20 09:30:00"
    assert "換算西元" in result.selected_fields["entry_time"].basis


def test_unrealized_pnl_is_not_misclassified_as_realized():
    result = parse_ocr_text("標的代號: BTCUSDT\n未實現損益: -125.5")

    assert result.value("pnl") is None
    assert result.value("unrealized_pnl") == -125.5
    assert any("不能當成已實現" in warning for warning in result.warnings)


def test_english_unrealized_pnl_is_not_reparsed_by_generic_pnl_rule():
    result = parse_ocr_text("Ticker: BTCUSDT\nUnrealized PnL: -125.5 USDT")

    assert result.value("pnl") is None
    assert result.value("unrealized_pnl") == -125.5
    assert any("不能當成已實現" in warning for warning in result.warnings)


def test_profit_label_cannot_override_an_explicit_negative_sign():
    result = parse_ocr_text("Ticker: AAPL\nProfit: -500 USD")

    assert result.value("pnl") is None
    candidate = result.candidates_for("pnl")[0]
    assert candidate.candidate_value == -500.0
    assert candidate.confidence < result.acceptance_threshold
    assert "正負號矛盾" in candidate.basis


def test_symbol_named_long_does_not_infer_position_side():
    result = parse_ocr_text(
        "Ticker: LONG\nEntry Price: 100\nExit Price: 90\nShares: 1"
    )

    assert result.value("symbol") == "LONG"
    assert result.value("side") is None
    assert "side" in result.missing_required_fields


def test_beginner_form_never_uses_low_confidence_as_value():
    result = parse_ocr_text("2330 台積電\n進場時間: 07/20 09:30\n損益: 500")
    form = build_beginner_form(result)

    assert _field(form, "symbol").status == "auto_filled"
    assert _field(form, "entry_time").status == "needs_review"
    assert _field(form, "entry_time").value is None
    assert _field(form, "entry_time").suggested_value == "07/20 09:30"
    assert _field(form, "pnl").value is None

    questions = build_review_questions(result)
    assert questions[0].required
    strategy_question = questions[-1]
    assert strategy_question.field == "strategy_reason"
    assert not strategy_question.required


def test_realized_pnl_currency_is_detected_but_still_requires_confirmation():
    result = parse_ocr_text("標的代號: AAPL\n已實現損益: +47.5 USD")

    pnl = result.candidates_for("pnl")[0]
    assert pnl.candidate_value == 47.5
    assert pnl.unit == "USD"
    currency_question = next(
        question for question in build_review_questions(result)
        if question.field == "pnl_currency"
    )
    assert currency_question.required
    assert "USD" in currency_question.reason


def test_custom_image_ocr_engine_keeps_core_dependency_optional():
    with tempfile.TemporaryDirectory() as directory:
        image = Path(directory) / "trade.png"
        image.write_bytes(b"not-a-real-image-custom-engine-does-not-open-it")

        seen: list[Path] = []

        def fake_ocr(path: Path) -> str:
            seen.append(path)
            return "標的代號: AAPL\n方向: 做多\n進場價: 200\n出場價: 210\n數量: 2 股"

        result = parse_screenshot_image(image, ocr_engine=fake_ocr)

        assert seen == [image]
        assert result.source == "image:trade.png"
        assert result.value("symbol") == "AAPL"
        assert result.value("entry_price") == 200.0
        assert result.warnings[0].startswith("OCR 可能讀錯")


if __name__ == "__main__":
    import traceback

    this_module = sys.modules[__name__]
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
        and getattr(value, "__module__", None) == this_module.__name__
    ]
    passed = failed = 0
    for test in tests:
        print(f"  RUN   {test.__name__}", flush=True)
        try:
            test()
            print(f"  PASS  {test.__name__}", flush=True)
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {test.__name__}", flush=True)
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
