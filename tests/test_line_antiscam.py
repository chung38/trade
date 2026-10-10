"""LINE 對話詐騙辨識的聚焦防退化測試。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.antiscam.text_scanner import render_scan, scan_text


def test_line_export_headers_and_split_phrase_keep_source_evidence():
    chat = """[LINE] 與投資老師的聊天記錄
儲存日期:2026/07/20 12:00
2026/07/20(一)
10:01\t林老師\t保 ○ 證
10:02\t林老師\t獲利，名額有限
10:03\t林老師\t請匯款到客服的私人帳戶
"""
    result = scan_text(chat)

    assert result.risk_level in ("高", "極高")
    assert any(h.category == "guaranteed_return" for h in result.hits)
    assert any(h.speaker == "林老師" and h.timestamp for h in result.hits)
    assert all("儲存日期" not in h.sentence for h in result.hits)


def test_line_messages_from_different_speakers_are_never_spliced():
    chat = """2026/07/20(一)
10:01\t阿明\t保證
10:02\t小美\t獲利
"""
    result = scan_text(chat)

    assert not any(h.category == "guaranteed_return" for h in result.hits)
    assert result.risk_level == "低"


def test_line_messages_on_different_days_are_never_spliced():
    chat = """2026/07/20(一)
23:59\t老師\t保證
2026/07/21(二)
00:01\t老師\t獲利
"""
    result = scan_text(chat)

    assert not any(h.category == "guaranteed_return" for h in result.hits)


def test_unicode_fullwidth_simplified_and_masked_words_are_normalized():
    result = scan_text("老师带单，保＊证＊获＊利，升级 Ｖ Ｉ Ｐ，私聊我")

    assert result.risk_level in ("高", "極高")
    categories = {h.category for h in result.hits}
    assert "guaranteed_return" in categories
    assert "fake_stock_group" in categories


def test_warning_only_applies_to_its_clause_not_later_transfer_order():
    result = scan_text("不要相信銀行客服，請立即匯款到老師的私人帳戶，解凍後才能出金")

    assert result.risk_level in ("高", "極高")
    assert any(h.category == "money_destination" for h in result.hits)


def test_multiline_warning_cannot_hide_a_later_transfer_order():
    result = scan_text("不要相信銀行客服\n現在請匯款到助理的私人帳戶")

    assert result.risk_level in ("高", "極高")
    assert any(h.category == "money_destination" for h in result.hits)


def test_warning_words_cannot_whitelist_scam_copy_or_a_different_target():
    samples = (
        "我的獨家投資手法保證獲利 現在就加入VIP",
        "不要相信銀行客服 現在請匯款到老師個人帳戶",
    )
    for text in samples:
        result = scan_text(text)
        assert result.risk_level in ("高", "極高"), text
        assert result.hits, text


def test_anti_scam_quote_and_quoted_self_denial_do_not_raise_risk():
    text = "新聞提醒:「保證獲利」與「我們不是詐騙」都是常見話術，請小心。"
    result = scan_text(text)

    assert result.risk_level == "低"
    assert not result.hits
    assert result.negated_hits


def test_quote_without_warning_context_is_still_evidence():
    result = scan_text("我們承諾「保證獲利」，現在就匯款到私人帳戶")

    assert result.risk_level in ("高", "極高")
    assert any(h.category == "guaranteed_return" for h in result.hits)


def test_plain_grammatical_negation_does_not_flag_guarantee():
    result = scan_text("本公司不保證獲利，也沒有穩賺不賠的投資，請先評估風險。")

    assert result.risk_level == "低"
    assert not result.hits
    assert result.negated_hits


def test_transfer_to_own_account_is_not_private_collection_evidence():
    result = scan_text("我從銀行匯款到自己的個人帳戶，用來整理每月預算。")

    assert result.risk_level not in ("高", "極高")
    assert not any(h.category == "money_destination" for h in result.hits)


def test_own_source_account_cannot_hide_a_private_destination():
    result = scan_text("請從自己的個人帳戶匯款到老師的個人帳戶")

    assert result.risk_level in ("高", "極高")
    assert any(h.category == "money_destination" for h in result.hits)


def test_withdrawal_prepayment_variants_are_high_risk():
    samples = (
        "提现前需要先交认证金，付完才能提现",
        "平台說先支付保證金才可提領本金",
        "出金通道凍結，需要補繳10%稅金才能解除風控",
    )
    for text in samples:
        result = scan_text(text)
        assert result.risk_level in ("高", "極高"), text
        assert any(h.category == "withdrawal_precondition" for h in result.hits), text


def test_legitimate_fee_deduction_is_not_prepayment_evidence():
    result = scan_text("券商公告:出金手續費會直接從款項扣除，不用另外匯款或先繳費。")

    assert result.risk_level not in ("高", "極高")
    assert not any(h.category == "withdrawal_precondition" for h in result.hits)


def test_split_withdrawal_demand_from_same_line_sender_is_composed():
    chat = """2026/07/20(一)
下午3:01\t平台客服\t你的出金通道已凍結
下午3:03\t平台客服\t需要先補繳認證金
下午3:04\t平台客服\t完成後才能提領
"""
    result = scan_text(chat)

    assert result.risk_level in ("高", "極高")
    composite = [h for h in result.hits if h.category == "withdrawal_precondition"]
    assert composite and composite[0].speaker == "平台客服"


def test_split_withdrawal_demand_survives_interleaving_and_four_messages():
    chats = (
        """2026/07/20(一)
10:00\t平台客服\t你要先申請出金
10:01\t小明\t為什麼
10:02\t平台客服\t必須先支付稅金
""",
        """2026/07/20(一)
10:00\t平台客服\t你要先申請出金
10:01\t平台客服\t請確認姓名
10:02\t平台客服\t請確認金額
10:03\t平台客服\t必須先支付稅金
""",
    )
    for chat in chats:
        result = scan_text(chat)
        assert result.risk_level in ("高", "極高")
        composite = [h for h in result.hits if h.category == "withdrawal_precondition"]
        assert composite and composite[0].speaker == "平台客服"


def test_render_is_ordinal_non_convicting_and_explainable():
    result = scan_text("10:01\t客服\t出金前需要先繳認證金，才能提領")
    report = render_scan(result)

    assert result.risk_level in ("高", "極高")
    assert "【可核對的對話證據】" in report
    assert "10:01 客服" in report
    assert "理由:" in report
    assert "詐騙機率" in report and "不產生" in report
    assert "判定為詐騙" not in report
    assert "一定是詐騙" not in report


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
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {test.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
