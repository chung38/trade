"""新手逐筆記錄與交易階段分流測試。

執行: python tests/test_onboarding.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cli import main  # noqa: E402
from core.ingest.loader import load_trades  # noqa: E402
from core.metrics.performance import compute_metrics  # noqa: E402
from core.report_html import _stage_colors  # noqa: E402
from core.models import Market  # noqa: E402
from core.verdict.judge import judge  # noqa: E402
from core.onboarding import (  # noqa: E402
    BEGINNER_COLUMNS,
    append_beginner_row,
    build_beginner_row,
    prompt_beginner_row,
    stage_from_analysis,
)


def _capture_cli(argv: list[str], *, stdin_text: str | None = None):
    """同時支援一般回傳與 argparse 的 SystemExit，供 CLI 契約測試。"""

    stdout = io.StringIO()
    stderr = io.StringIO()
    original_stdin = sys.stdin
    if stdin_text is not None:
        sys.stdin = io.StringIO(stdin_text)
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(argv)
            except SystemExit as exc:
                code = int(exc.code)
    finally:
        sys.stdin = original_stdin
    return code, stdout.getvalue(), stderr.getvalue()


def _result(*, n=30, expectancy=10.0, discourage=False, persisted=False):
    return SimpleNamespace(
        metrics=SimpleNamespace(
            total_trades=n,
            expectancy=expectancy,
            return_metrics_reliable=True,
            drawdown_pct_reliable=True,
            currency_reliable=True,
        ),
        verdict=SimpleNamespace(should_discourage=discourage, headline="測試裁決"),
        out_of_sample=SimpleNamespace(edge_persisted=persisted),
    )


def test_beginner_row_accepts_pnl_only():
    row = build_beginner_row(
        symbol="2330", pnl="-1,250 TWD", exit_time="2026-07-20", strategy="突破失敗"
    )
    assert row["損益"] == "-1250"
    assert row["進場價"] == ""
    assert row["進場時間"] == ""
    assert row["出場時間"] == "2026-07-20"
    assert row["方向"] == ""
    accounting = build_beginner_row(symbol="2330", pnl="(NT$1,250)")
    assert accounting["損益"] == "-1250"


def test_beginner_row_requires_pnl_or_complete_prices():
    try:
        build_beginner_row(symbol="AAPL", entry_price="100")
    except ValueError as exc:
        assert "進場價、出場價、數量" in str(exc)
    else:
        raise AssertionError("不完整價量不可被接受")

    try:
        build_beginner_row(
            symbol="AAPL", entry_price="100", exit_price="110", quantity="1"
        )
    except ValueError as exc:
        assert "必須明示方向" in str(exc)
    else:
        raise AssertionError("價差推算不可把空白方向默認成做多")


def test_interactive_record_has_no_default_long_direction():
    answers = iter(["AAPL", "", "25 USD", "2026-07-20", "突破"])
    prompts: list[str] = []

    row = prompt_beginner_row(
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        output_fn=lambda _message: None,
    )

    assert row["方向"] == ""
    assert any("無預設" in prompt for prompt in prompts)
    assert all("預設買" not in prompt for prompt in prompts)


def test_beginner_row_rejects_bad_direction_and_time_order():
    for kwargs in (
        {"symbol": "2330", "pnl": "10 TWD", "side": "看多"},
        {
            "symbol": "2330", "pnl": "10 TWD",
            "entry_time": "2026-07-21", "exit_time": "2026-07-20",
        },
    ):
        try:
            build_beginner_row(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("無效輸入不可被接受")


def test_beginner_row_escapes_spreadsheet_formula():
    row = build_beginner_row(symbol="2330", pnl="10 TWD", strategy="=HYPERLINK(\"x\")")
    assert row["策略"].startswith("'=")


def test_append_then_loader_reads_pnl_only_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "my_trades.csv"
        row1 = build_beginner_row(symbol="2330", pnl="100 TWD", strategy="月線")
        row2 = build_beginner_row(symbol="2330", pnl="-50 TWD", strategy="月線")
        _, count1 = append_beginner_row(path, row1)
        _, count2 = append_beginner_row(path, row2)
        log = load_trades(path)
        assert count1 == 1 and count2 == 2
        assert len(log.trades) == 2
        assert [t.pnl for t in log.trades] == [100.0, -50.0]
        assert all(not t.entry_time_known and t.exit_time_known for t in log)
        assert all(not t.is_day_trade and t.holding_days is None for t in log)


def test_append_refuses_unknown_existing_layout():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "other.csv"
        path.write_text("foo,bar\n1,2\n", encoding="utf-8")
        row = build_beginner_row(symbol="2330", pnl="100 TWD")
        try:
            append_beginner_row(path, row)
        except ValueError as exc:
            assert "避免寫壞" in str(exc)
        else:
            raise AssertionError("不可猜測並改寫陌生 CSV")


def test_forex_price_pnl_rejects_ambiguous_lot_quantity():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fx.csv"
        path.write_text(
            "symbol,side,entry_price,exit_price,lots\nEURUSD,long,1.10,1.11,1\n",
            encoding="utf-8",
        )
        try:
            load_trades(path, auto_estimate_costs=False)
        except ValueError as exc:
            assert "外匯價差損益" in str(exc) and "直接" in str(exc)
        else:
            raise AssertionError("外匯手數不可被當成基礎貨幣單位")


def test_forex_explicit_units_still_require_broker_converted_pnl():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fx_units.csv"
        path.write_text(
            "symbol,side,entry_price,exit_price,units\nEURUSD,long,1.10,1.11,100000\n",
            encoding="utf-8",
        )
        try:
            load_trades(path, auto_estimate_costs=False)
        except ValueError as exc:
            assert "帳戶換匯口徑未知" in str(exc)
        else:
            raise AssertionError("外匯價差不可跨報價幣直接推算為帳戶損益")


def test_forex_direct_pnl_kept_but_ambiguous_notional_discarded():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fx_pnl.csv"
        path.write_text(
            "symbol,entry_price,exit_price,lots,net_pnl,pnl_currency\n"
            "EURUSD,1.10,1.11,1,800,USD\n",
            encoding="utf-8",
        )
        log = load_trades(path)
        assert log.trades[0].pnl == 800.0
        assert log.trades[0].contract_value == 0.0
        assert "數量單位不明" in log.source


def test_beginner_forex_requires_account_currency_and_rejects_price_only():
    for kwargs, expected in (
        ({"symbol": "EURUSD", "pnl": "100"}, "帳戶"),
        ({"symbol": "EURUSD", "entry_price": 1.1, "exit_price": 1.11,
          "quantity": 100000}, "外匯不可只用價差"),
    ):
        try:
            build_beginner_row(**kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("外匯新手紀錄不可猜幣別或換匯口徑")

    row = build_beginner_row(symbol="EURUSD", pnl="100 USD")
    assert row["損益"] == "100" and row["損益幣別"] == "USD"


def test_beginner_currency_amount_parsing_supports_prefix_and_suffix_codes():
    cases = (
        ("-120 JPY", "-120", "JPY"),
        ("JPY -120", "-120", "JPY"),
        ("100 USDT", "100", "USDT"),
        ("US$100", "100", "USD"),
        ("25 BUSD", "25", "BUSD"),
    )
    for raw, expected_amount, expected_currency in cases:
        row = build_beginner_row(symbol="BTCUSDT", pnl=raw)
        assert row["損益"] == expected_amount
        assert row["損益幣別"] == expected_currency


def test_append_refuses_to_drop_currency_into_legacy_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.csv"
        legacy_columns = [column for column in BEGINNER_COLUMNS if column != "損益幣別"]
        path.write_text(",".join(legacy_columns) + "\n", encoding="utf-8-sig")
        row = build_beginner_row(symbol="2330", pnl="100 TWD")

        try:
            append_beginner_row(path, row)
        except ValueError as exc:
            assert "缺少『損益幣別』" in str(exc)
        else:
            raise AssertionError("不得把幣別靜默丟進舊格式")


def test_mixed_pnl_currencies_are_rejected_instead_of_added():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mixed.csv"
        path.write_text(
            "symbol,net_pnl,pnl_currency\n2330,1000,TWD\nAAPL,100,USD\n",
            encoding="utf-8",
        )
        try:
            load_trades(path)
        except ValueError as exc:
            assert "不同或不明的損益幣別" in str(exc)
            assert "TWD" in str(exc) and "USD" in str(exc)
        else:
            raise AssertionError("TWD 與 USD 不可靜默相加")


def test_malformed_direct_pnl_never_falls_back_to_price_difference():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_pnl.csv"
        path.write_text(
            "symbol,side,entry_price,exit_price,quantity,net_pnl,pnl_currency\n"
            "2330,long,100,110,1000,1O0,TWD\n",
            encoding="utf-8",
        )
        try:
            load_trades(path, auto_estimate_costs=False)
        except ValueError as exc:
            assert "拒絕忽略它後改用價差" in str(exc)
        else:
            raise AssertionError("錯字 pnl 不得被價差結果取代")


def test_pnl_amount_and_currency_column_must_not_conflict():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "currency_conflict.csv"
        path.write_text(
            "symbol,net_pnl,pnl_currency\nAAPL,100 TWD,USD\n",
            encoding="utf-8",
        )
        try:
            load_trades(path)
        except ValueError as exc:
            assert "幣別衝突" in str(exc)
        else:
            raise AssertionError("金額與欄位幣別衝突時不可擇一猜測")

    try:
        build_beginner_row(symbol="AAPL", pnl="100 TWD", currency="USD")
    except ValueError as exc:
        assert "幣別" in str(exc) and "TWD" in str(exc) and "USD" in str(exc)
    else:
        raise AssertionError("新手輸入的兩個幣別來源也必須一致")


def test_cny_and_cnh_are_not_silently_merged():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cny_cnh.csv"
        path.write_text(
            "symbol,net_pnl,pnl_currency\nAAPL,100,CNY\nAAPL,100,CNH\n",
            encoding="utf-8",
        )
        try:
            load_trades(path)
        except ValueError as exc:
            assert "不同或不明的損益幣別" in str(exc)
            assert "CNY" in str(exc) and "CNH" in str(exc)
        else:
            raise AssertionError("在岸與離岸人民幣不可無匯率直接相加")


def test_ambiguous_or_gross_pnl_headers_cannot_masquerade_as_net():
    with tempfile.TemporaryDirectory() as tmp:
        ambiguous = Path(tmp) / "ambiguous.csv"
        ambiguous.write_text(
            "symbol,盈虧,pnl_currency\nAAPL,100,USD\n",
            encoding="utf-8",
        )
        try:
            load_trades(ambiguous)
        except ValueError as exc:
            assert "gross" in str(exc) and "net_pnl" in str(exc)
        else:
            raise AssertionError("模糊 pnl header 必須要求確認淨額語意")

        for header in ("realized_pnl", "已實現損益"):
            uncertain = Path(tmp) / f"uncertain_{len(header)}.csv"
            uncertain.write_text(
                f"symbol,{header},pnl_currency\nAAPL,100,USD\n",
                encoding="utf-8",
            )
            try:
                load_trades(uncertain)
            except ValueError as exc:
                assert "無法確認" in str(exc) and "--field" in str(exc)
            else:
                raise AssertionError(f"{header} 未明示淨額，不得自動當成 net pnl")

        gross = Path(tmp) / "gross.csv"
        gross.write_text(
            "symbol,profit,pnl_currency,total_fee\nAAPL,100,USD,10\n",
            encoding="utf-8",
        )
        trade = load_trades(gross).trades[0]
        assert trade.pnl == 90
        assert trade.fees == 10


def test_market_hint_prevents_symbol_shaped_multiplier_leak():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "forced_us.csv"
        path.write_text(
            "symbol,side,entry_price,exit_price,quantity\nTXF,long,100,110,1\n",
            encoding="utf-8",
        )
        log = load_trades(
            path, market_hint=Market.US_STOCK, auto_estimate_costs=False
        )
        assert log.trades[0].market == Market.US_STOCK
        assert log.trades[0].contract_multiplier == 1.0
        assert log.trades[0].pnl == 10.0


def test_unknown_derivative_multiplier_disables_notional_risk_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "unknown_future.csv"
        path.write_text(
            "symbol,entry_time,exit_time,entry_price,exit_price,quantity,net_pnl,pnl_currency\n"
            "ZZZF202606,2026-01-01,2026-01-02,100,110,1,5000,TWD\n"
            "ZZZF202606,2026-01-03,2026-01-04,100,90,1,-4000,TWD\n",
            encoding="utf-8",
        )
        log = load_trades(path, market_hint=Market.TW_FUTURES)

    assert all(not t.contract_multiplier_known and not t.notional_reliable for t in log)
    assert all(t.contract_value == 0 and t.return_pct == 0 for t in log)
    metrics = compute_metrics(log)
    assert not metrics.return_metrics_reliable
    assert not metrics.drawdown_pct_reliable
    verdict = judge(log, metrics=metrics, n_bootstrap=50)
    assert all(flag.code != "severe_drawdown" for flag in verdict.red_flags)


def test_direct_pnl_currency_cannot_be_divided_by_foreign_notional():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "us_in_twd.csv"
        path.write_text(
            "symbol,entry_time,exit_time,entry_price,exit_price,quantity,net_pnl,pnl_currency\n"
            "AAPL,2026-01-01,2026-01-02,100,101,1,3200,TWD\n"
            "AAPL,2026-01-03,2026-01-04,100,99,1,-1600,TWD\n",
            encoding="utf-8",
        )
        log = load_trades(path)

    assert all(not t.notional_reliable and t.return_pct == 0 for t in log)
    metrics = compute_metrics(log)
    assert metrics.pnl_currency == "TWD"
    assert not metrics.return_metrics_reliable
    assert not metrics.drawdown_pct_reliable


def test_direct_pnl_is_treated_as_net_and_never_charged_estimated_fees_again():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "net_pnl.csv"
        path.write_text(
            "symbol,entry_price,exit_price,quantity,net_pnl,pnl_currency\n"
            "2330,100,101,1000,100,TWD\n",
            encoding="utf-8",
        )
        log = load_trades(path)

    assert log.trades[0].pnl == 100.0
    assert log.trades[0].fees == 0.0
    assert "已扣手續費/稅/滑價的淨損益" in log.source


def test_direct_net_pnl_keeps_bad_times_but_marks_each_field_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_times_net.csv"
        path.write_text(
            "symbol,entry_time,exit_time,entry_price,quantity,net_pnl,pnl_currency\n"
            "AAPL,not-a-time,2026-07-02,100,1,-25,USD\n"
            "AAPL,2026-07-03,also-bad,100,1,10,USD\n",
            encoding="utf-8",
        )
        log = load_trades(path)

        assert [trade.pnl for trade in log] == [-25.0, 10.0]
        assert not log.trades[0].entry_time_known
        assert log.trades[0].exit_time_known
        assert log.trades[1].entry_time_known
        assert not log.trades[1].exit_time_known
        assert "時間格式錯誤但因 direct net pnl 保留" in log.source

        price_path = Path(tmp) / "bad_time_prices.csv"
        price_path.write_text(
            "symbol,side,entry_time,exit_time,entry_price,exit_price,quantity\n"
            "AAPL,long,not-a-time,2026-07-02,100,110,1\n",
            encoding="utf-8",
        )
        try:
            load_trades(price_path, auto_estimate_costs=False)
        except ValueError as exc:
            assert "進場時間格式無法解析" in str(exc)
        else:
            raise AssertionError("價差推算仍須拒絕無法解析的時間")


def test_direct_pnl_with_invalid_or_missing_basis_cannot_claim_return():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_basis.csv"
        path.write_text(
            "symbol,entry_price,quantity,net_pnl,pnl_currency\n"
            "AAPL,0,1,-10,USD\n"
            "AAPL,100,0,20,USD\n"
            "AAPL,,1,-5,USD\n"
            "AAPL,100,,5,USD\n",
            encoding="utf-8",
        )
        log = load_trades(path)

    assert [trade.pnl for trade in log] == [-10.0, 20.0, -5.0, 5.0]
    assert all(not trade.notional_reliable for trade in log)
    assert all(trade.contract_value == 0 and trade.return_pct == 0 for trade in log)
    assert "缺少有效進場價或數量" in log.source


def test_bad_optional_fee_does_not_select_out_direct_net_pnl():
    with tempfile.TemporaryDirectory() as tmp:
        net = Path(tmp) / "bad_net_fees.csv"
        net.write_text(
            "symbol,net_pnl,pnl_currency,total_fee\n"
            "AAPL,-50,USD,not-a-fee\n"
            "AAPL,25,USD,-3\n",
            encoding="utf-8",
        )
        log = load_trades(net)
        assert [trade.pnl for trade in log] == [-50.0, 25.0]
        assert all(trade.fees == 0 for trade in log)
        assert "optional fee 無法解析或為負數" in log.source

        gross = Path(tmp) / "bad_gross_fee.csv"
        gross.write_text(
            "symbol,profit,pnl_currency,total_fee\nAAPL,100,USD,not-a-fee\n",
            encoding="utf-8",
        )
        try:
            load_trades(gross)
        except ValueError as exc:
            assert "成本欄" in str(exc) and "無法解析" in str(exc)
        else:
            raise AssertionError("gross pnl 的成本不可信時仍必須拒絕")


def test_canonical_pnl_header_keeps_minimal_schema_compatibility():
    """標準欄名 pnl 本身就是 direct net PnL 契約，不需多餘價量欄。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "canonical.csv"
        path.write_text(
            "symbol,pnl,pnl_currency\nAAPL,125,USD\n",
            encoding="utf-8",
        )
        trade = load_trades(path).trades[0]

    assert trade.pnl == 125.0
    assert trade.pnl_is_direct
    assert trade.pnl_currency == "USD"

    with tempfile.TemporaryDirectory() as tmp:
        explicit_net = Path(tmp) / "explicit_net.csv"
        explicit_net.write_text(
            "symbol,已實現淨損益,pnl_currency\n2330,-80,TWD\n",
            encoding="utf-8",
        )
        explicit_trade = load_trades(explicit_net).trades[0]
    assert explicit_trade.pnl == -80.0
    assert explicit_trade.pnl_is_direct


def test_explicit_pnl_field_override_confirms_custom_net_column():
    """--field pnl=... 是使用者明確確認該自訂欄為已扣成本淨損益。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "custom.csv"
        path.write_text(
            "symbol,我的淨利,結算幣別\n2330,-80,TWD\n",
            encoding="utf-8",
        )
        trade = load_trades(
            path,
            field_overrides={"pnl": "我的淨利"},
        ).trades[0]

    assert trade.pnl == -80.0
    assert trade.pnl_is_direct
    assert trade.pnl_currency == "TWD"


def test_stage_never_approves_small_sample_or_failed_oos():
    assert stage_from_analysis(_result(n=29)).code == "paper_only"
    assert stage_from_analysis(_result(n=5, expectancy=-1, discourage=True)).code == "stop_real_money"
    assert stage_from_analysis(_result(n=60, discourage=True)).code == "stop_real_money"
    assert stage_from_analysis(_result(n=60, persisted=False)).code == "paper_until_oos"


def test_risk_data_stage_uses_same_warning_color_as_other_paper_stages():
    assert _stage_colors("paper_until_risk_data") == _stage_colors("paper_until_oos")


def test_stage_requires_reliable_account_drawdown_basis():
    result = _result(n=60, persisted=True)
    result.metrics.drawdown_pct_reliable = False
    assert stage_from_analysis(result).code == "paper_until_risk_data"


def test_price_difference_requires_side_but_direct_pnl_can_mark_it_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        price_path = Path(tmp) / "missing_side.csv"
        price_path.write_text(
            "symbol,entry_price,exit_price,quantity\nAAPL,100,90,1\n",
            encoding="utf-8",
        )
        try:
            load_trades(price_path, auto_estimate_costs=False)
        except ValueError as exc:
            assert "做多/做空方向" in str(exc)
        else:
            raise AssertionError("價差損益不可默認做多")

        direct_path = Path(tmp) / "direct.csv"
        direct_path.write_text(
            "symbol,net_pnl,pnl_currency\nAAPL,10,USD\n",
            encoding="utf-8",
        )
        trade = load_trades(direct_path).trades[0]
        assert trade.pnl == 10 and not trade.side_known


def test_short_price_difference_requires_broker_net_pnl():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "short.csv"
        path.write_text(
            "symbol,side,entry_price,exit_price,quantity\nAAPL,short,100,90,10\n",
            encoding="utf-8",
        )
        try:
            load_trades(path)
        except ValueError as exc:
            assert "借券費" in str(exc) and "direct net pnl" in str(exc)
        else:
            raise AssertionError("放空價差不得漏算借券成本")


def test_short_buy_sell_headers_are_not_silently_reversed():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "short_future.csv"
        path.write_text(
            "symbol,side,buy_price,sell_price,quantity\n"
            "TXFG5,short,90,100,1\n",
            encoding="utf-8",
        )
        try:
            load_trades(path, market_hint=Market.TW_FUTURES, auto_estimate_costs=False)
        except ValueError as exc:
            assert "不可把 buy/sell" in str(exc)
        else:
            raise AssertionError("空單 buy/sell 欄不可固定映射成進出場")


def test_direct_pnl_generic_trade_price_never_becomes_entry_basis():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "close_fill.csv"
        path.write_text(
            "symbol,price,quantity,net_pnl,pnl_currency\n"
            "AAPL,110,10,100,USD\n",
            encoding="utf-8",
        )
        log = load_trades(path)
        trade = log.trades[0]
        assert not trade.notional_reliable
        assert trade.return_pct == 0
        assert "無法證明是實際進場成本" in log.source


def test_fee_and_tax_components_are_summed_without_dropping_one():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "components.csv"
        path.write_text(
            "symbol,side,entry_price,exit_price,quantity,手續費,證交稅\n"
            "2330,long,100,101,1000,100,300\n",
            encoding="utf-8",
        )
        trade = load_trades(path, auto_estimate_costs=False).trades[0]
        assert trade.fees == 400
        assert trade.pnl == 600

        partial_row = Path(tmp) / "partial_row.csv"
        partial_row.write_text(
            "symbol,side,entry_price,exit_price,quantity,手續費,證交稅\n"
            "2330,long,100,101,1000,100,\n",
            encoding="utf-8",
        )
        try:
            load_trades(partial_row, auto_estimate_costs=False)
        except ValueError as exc:
            assert "部分成本冒充完整成本" in str(exc)
        else:
            raise AssertionError("每一列都必須有完整成本分項")

        tax_only = Path(tmp) / "tax_only.csv"
        tax_only.write_text(
            "symbol,side,entry_price,exit_price,quantity,證交稅\n"
            "2330,long,100,101,1000,300\n",
            encoding="utf-8",
        )
        try:
            load_trades(tax_only, auto_estimate_costs=False)
        except ValueError as exc:
            assert "部分成本冒充完整成本" in str(exc)
        else:
            raise AssertionError("只有稅不可視為完整交易成本")

        commission_only = Path(tmp) / "commission_only.csv"
        commission_only.write_text(
            "symbol,side,entry_price,exit_price,quantity,手續費\n"
            "2330,long,100,101,1000,100\n",
            encoding="utf-8",
        )
        try:
            load_trades(commission_only, auto_estimate_costs=False)
        except ValueError as exc:
            assert "部分成本冒充完整成本" in str(exc)
        else:
            raise AssertionError("台股只有佣金也不可視為完整交易成本")


def test_settlement_cashflow_is_never_treated_as_realized_pnl():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cashflow.csv"
        path.write_text(
            "symbol,淨收付,損益幣別\n2330,100000,TWD\n",
            encoding="utf-8",
        )
        try:
            load_trades(path)
        except ValueError as exc:
            assert "含本金" in str(exc) and "不是已平倉實現損益" in str(exc)
        else:
            raise AssertionError("交割收付款不可冒充獲利")


def test_lot_column_only_multiplies_taiwan_stock_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mixed_units.csv"
        path.write_text(
            "symbol,entry_price,exit_price,張數,net_pnl,pnl_currency\n"
            "2330,100,101,1,100,TWD\n"
            "AAPL,100,101,1,100,TWD\n",
            encoding="utf-8",
        )
        log = load_trades(path)
        assert log.trades[0].quantity == 1000
        assert log.trades[1].quantity == 0
        assert not log.trades[1].notional_reliable
        assert "不是可安全套用" in log.source


def test_english_lots_mean_taiwan_board_lots_but_hands_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        lots = Path(tmp) / "lots.csv"
        lots.write_text(
            "symbol,side,entry_price,exit_price,lots\n2330,long,100,101,1\n",
            encoding="utf-8",
        )
        trade = load_trades(lots, auto_estimate_costs=False).trades[0]
        assert trade.quantity == 1000 and trade.pnl == 1000

        hands = Path(tmp) / "hands.csv"
        hands.write_text(
            "symbol,side,entry_price,exit_price,手數\n2330,long,100,101,1\n",
            encoding="utf-8",
        )
        try:
            load_trades(hands, auto_estimate_costs=False)
        except ValueError as exc:
            assert "無法對 tw_stock 安全換算" in str(exc)
        else:
            raise AssertionError("手數不可默認成一股或一張")


def test_stage_only_allows_tiny_validation_after_oos_persists():
    stage = stage_from_analysis(_result(n=60, persisted=True))
    assert stage.code == "tiny_live_validation"
    assert "重押" in stage.title


def test_cli_start_and_noninteractive_record():
    assert main(["start"]) == 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mine.csv"
        rc = main([
            "record", "--out", str(path), "--symbol", "AAPL",
            "--pnl", "25", "--currency", "USD", "--exit-time", "2026-07-20",
            "--strategy", "突破",
        ])
        assert rc == 0
        assert path.exists()
        assert not load_trades(path).trades[0].side_known


def test_cli_scan_screenshot_from_ocr_text_writes_review_json():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "review.json"
        rc = main([
            "scan-screenshot", "--text",
            "股票代號: 2330\n方向: 做多\n進場價: 980\n出場價: 1000\n"
            "數量: 1000 股\n進場理由: 突破月線",
            "--json", str(out),
        ])
        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["safe_fields"]["symbol"]["value"] == "2330"
        assert any(clue["code"] == "breakout" for clue in payload["strategy_clues"])
        assert "strategy_notice" in payload
        assert "不是 OCR 正確率" in payload["confidence_notice"]
        assert payload["needs_manual_review"] is True
        assert payload["requires_human_review"] is True


def test_cli_record_pnl_help_requires_explicit_currency():
    code, stdout, stderr = _capture_cli(["record", "--help"])
    assert code == 0
    assert stderr == ""
    assert "必須在數值內帶幣別" in stdout
    assert "--currency" in stdout


def test_cli_scan_screenshot_requires_exactly_one_source():
    invalid_argv = (
        ["scan-screenshot"],
        ["scan-screenshot", "trade.png", "--text", "OCR 文字"],
        ["scan-screenshot", "trade.png", "--text-file", "ocr.txt"],
        ["scan-screenshot", "--text-file", "ocr.txt", "--text", "OCR 文字"],
    )
    for argv in invalid_argv:
        code, _stdout, stderr = _capture_cli(argv)
        assert code == 2
        assert "必須且只能提供一個來源" in stderr
        assert "IMAGE、--text-file PATH 或 --text TEXT" in stderr

    code, stdout, stderr = _capture_cli(["scan-screenshot", "--help"])
    assert code == 0
    assert stderr == ""
    assert "輸入來源三選一" in stdout
    assert "必須只提供一個" in stdout


def test_cli_scan_text_rejects_conflicting_sources_and_keeps_stdin():
    code, _stdout, stderr = _capture_cli(
        ["scan-text", "直接文字", "--file", "chat.txt"]
    )
    assert code == 2
    assert "文字參數與 --file 不可同時使用" in stderr

    code, stdout, stderr = _capture_cli(["scan-text", "--help"])
    assert code == 0
    assert stderr == ""
    assert "直接文字與 --file 不可同時使用" in stdout
    assert "stdin" in stdout

    code, stdout, stderr = _capture_cli(
        ["scan-text"], stdin_text="今天只是一般風險教育與投資紀錄討論。"
    )
    assert code == 0
    assert stderr == ""
    assert stdout


if __name__ == "__main__":
    import traceback

    module = sys.modules[__name__]
    tests = [
        value for name, value in sorted(vars(module).items())
        if name.startswith("test_") and callable(value)
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
