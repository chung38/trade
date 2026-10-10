"""缺少交易時間時不得假裝當沖或時序資料的聚焦防退化測試。"""

from __future__ import annotations

import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyzer import analyze_file  # noqa: E402
from core.ingest.costs import estimate_round_trip_cost  # noqa: E402
from core.ingest.loader import _parse_time, load_trades  # noqa: E402
from core.metrics.performance import compute_metrics  # noqa: E402
from core.models import Market, Side, Trade, TradeLog  # noqa: E402
from core.report_html import render_html_report  # noqa: E402
from core.strategy.profiler import profile_strategy  # noqa: E402
from core.trend.timeline import (  # noqa: E402
    analyze_trend,
    bucket_by_period,
    detect_decay,
    equity_curve,
    render_trend_text,
    rolling_expectancy,
)


def _trade(
    index: int = 0,
    *,
    days: int = 0,
    entry_known: bool = True,
    exit_known: bool = True,
    pnl: float = 10.0,
) -> Trade:
    entry = datetime(2025, 1, 1) + timedelta(days=index)
    exit_time = entry + timedelta(days=days) if exit_known else entry
    return Trade(
        symbol="2330",
        market=Market.TW_STOCK,
        side=Side.LONG,
        entry_time=entry if entry_known else datetime(1970, 1, 1),
        exit_time=exit_time,
        entry_price=100.0,
        exit_price=101.0,
        quantity=1000.0,
        pnl=pnl,
        entry_time_known=entry_known,
        exit_time_known=exit_known,
    )


def _load_csv(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trades.csv"
        path.write_text(text, encoding="utf-8")
        return load_trades(path)


def test_trade_defaults_preserve_existing_constructors():
    trade = _trade()
    assert trade.entry_time_known and trade.exit_time_known
    assert trade.is_day_trade
    assert trade.holding_days == 0.0


def test_offset_timestamps_preserve_absolute_order_in_utc():
    earlier = _parse_time("2026-01-01T10:00:00+08:00")
    later = _parse_time("2026-01-01T03:00:00+00:00")

    assert earlier == datetime(2026, 1, 1, 2, 0)
    assert later == datetime(2026, 1, 1, 3, 0)
    assert earlier < later


def test_utc_sorting_does_not_change_local_calendar_day_semantics():
    crossed = _load_csv(
        "symbol,side,entry_time,exit_time,entry_price,exit_price,quantity\n"
        "2330,long,2026-01-01T23:30:00+08:00,2026-01-02T00:30:00+08:00,100,101,1000\n"
    ).trades[0]
    same_local_day = _load_csv(
        "symbol,side,entry_time,exit_time,entry_price,exit_price,quantity\n"
        "2330,long,2026-01-02T00:30:00+08:00,2026-01-02T23:30:00+08:00,100,101,1000\n"
    ).trades[0]

    assert not crossed.is_day_trade
    assert same_local_day.is_day_trade


def test_unknown_time_is_not_day_trade_or_holding_period():
    trade = _trade(entry_known=False, exit_known=False)
    assert not trade.is_day_trade
    assert trade.holding_days is None


def test_loader_marks_pnl_only_time_as_unknown_and_discloses_it():
    log = _load_csv(
        "symbol,net_pnl,pnl_currency\n2330,100,TWD\n2330,-50,TWD\n"
    )

    assert len(log.trades) == 2
    assert all(not t.entry_time_known and not t.exit_time_known for t in log)
    assert all(not t.is_day_trade and t.holding_days is None for t in log)
    assert "時間資料不完整" in log.source
    assert "缺出場 2 筆" in log.source


def test_missing_exit_time_never_receives_day_trade_cost_discount():
    log = _load_csv(
        "symbol,side,entry_time,entry_price,exit_price,quantity\n"
        "2330,long,2025-01-02 09:00,100,101,1000\n"
    )
    trade = log.trades[0]
    expected = estimate_round_trip_cost(
        Market.TW_STOCK,
        Side.LONG,
        100.0,
        101.0,
        1000.0,
        is_day_trade=False,
    )

    assert trade.entry_time_known and not trade.exit_time_known
    assert not trade.is_day_trade
    assert abs(trade.fees - expected) < 1e-9


def test_full_times_still_use_day_trade_cost_and_semantics():
    log = _load_csv(
        "symbol,side,entry_time,exit_time,entry_price,exit_price,quantity\n"
        "2330,long,2025-01-02 09:00,2025-01-02 13:00,100,101,1000\n"
    )
    trade = log.trades[0]
    expected = estimate_round_trip_cost(
        Market.TW_STOCK,
        Side.LONG,
        100.0,
        101.0,
        1000.0,
        is_day_trade=True,
    )

    assert trade.entry_time_known and trade.exit_time_known and trade.is_day_trade
    assert abs(trade.fees - expected) < 1e-9


def test_metrics_all_unknown_do_not_claim_intraday_style():
    metrics = compute_metrics(TradeLog([_trade(i, entry_known=False, exit_known=False) for i in range(5)]))

    assert metrics.timing_known_trades == 0
    assert metrics.timing_unknown_trades == 5
    assert not metrics.timing_data_complete
    assert not metrics.timing_metrics_available
    assert metrics.intraday_ratio == 0.0
    assert not metrics.is_mostly_intraday
    assert "0/5" in metrics.timing_note


def test_metrics_partial_time_use_only_known_subset_and_withhold_global_label():
    known = _trade(1, days=2)
    unknown = _trade(2, entry_known=False, exit_known=False)
    metrics = compute_metrics(TradeLog([known, unknown]))

    assert metrics.timing_known_trades == 1
    assert metrics.timing_unknown_trades == 1
    assert metrics.avg_holding_days == 2.0
    assert metrics.intraday_ratio == 0.0
    assert not metrics.is_mostly_intraday
    assert not metrics.sequence_metrics_reliable
    assert metrics.max_drawdown == 0.0
    assert metrics.max_consecutive_losses == 0
    assert "1/2" in metrics.timing_note


def test_profiler_all_unknown_reports_timing_unavailable():
    profile = profile_strategy(
        TradeLog([_trade(i, entry_known=False, exit_known=False) for i in range(3)])
    )

    assert profile.style_code == "timing_unavailable"
    assert not profile.timing_metrics_available
    assert profile.timing_unknown_trades == 3
    assert any("不得把佔位日期當成當沖" in note for note in profile.notes)


def test_profiler_partial_time_discloses_subset():
    profile = profile_strategy(
        TradeLog([_trade(1, days=3), _trade(2, entry_known=False, exit_known=False)])
    )

    assert profile.avg_holding_days == 3.0
    assert profile.timing_known_trades == 1
    assert profile.timing_unknown_trades == 1
    assert not profile.timing_data_complete
    assert profile.style_code == "timing_incomplete"
    assert any("已知子集" in note for note in profile.notes)


def test_unknown_exit_time_never_reorders_or_invents_sequence_risk():
    first = _trade(0, pnl=-100)
    unknown = _trade(1, entry_known=True, exit_known=False, pnl=100)
    last = _trade(2, pnl=-100)
    metrics = compute_metrics(TradeLog([first, unknown, last]))

    assert not metrics.sequence_metrics_reliable
    assert not metrics.drawdown_pct_reliable
    assert metrics.max_drawdown == 0.0
    assert metrics.max_consecutive_losses == 0


def test_identical_exit_timestamps_do_not_invent_intraperiod_sequence():
    first = _trade(1, pnl=100)
    second = _trade(2, pnl=-200)
    second.entry_time = first.entry_time
    second.exit_time = first.exit_time

    metrics = compute_metrics(TradeLog([first, second]))

    assert not metrics.sequence_metrics_reliable
    assert metrics.max_drawdown == 0
    assert metrics.max_consecutive_losses == 0
    assert "相同出場時間" in metrics.sequence_note
    assert "累積曲線無法計算" in metrics.sequence_note


def test_trend_report_is_explicitly_unavailable_for_any_missing_exit_time():
    log = TradeLog([_trade(1), _trade(2, entry_known=False, exit_known=False)])
    report = analyze_trend(log, rolling_window=2)
    rendered = render_trend_text(report)

    assert not report.available
    assert report.buckets == [] and report.equity == [] and report.rolling == []
    assert not report.decay.enough_data and report.decay.direction == "unknown"
    assert "1970" in report.unavailable_reason
    assert "列順序" in report.unavailable_reason
    assert "時間趨勢不可用" in rendered
    assert report.as_dict()["available"] is False


def test_low_level_trend_functions_refuse_unknown_exit_time():
    log = TradeLog([_trade(entry_known=False, exit_known=False)])
    for fn, kwargs in (
        (bucket_by_period, {}),
        (equity_curve, {}),
        (rolling_expectancy, {"window": 2}),
    ):
        try:
            fn(log, **kwargs)
        except ValueError as exc:
            assert "時間趨勢不可用" in str(exc)
        else:
            raise AssertionError(f"{fn.__name__} 應拒絕缺少出場時間的資料")


def test_exit_only_data_can_drive_trend_but_not_holding_style():
    trades = [
        _trade(i, entry_known=False, exit_known=True, pnl=float(i + 1))
        for i in range(4)
    ]
    log = TradeLog(trades)
    report = analyze_trend(log, rolling_window=2)
    profile = profile_strategy(log)

    assert report.available and report.buckets and report.equity
    assert profile.style_code == "timing_unavailable"
    assert not profile.timing_metrics_available


def test_reports_disclose_unavailable_timing_instead_of_printing_zero_days():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pnl_only.csv"
        path.write_text(
            "symbol,net_pnl,pnl_currency\n2330,100,TWD\n2330,-50,TWD\n",
            encoding="utf-8",
        )
        result = analyze_file(path, n_bootstrap=50)

    trend = analyze_trend(result.log, rolling_window=2)
    html = render_html_report(result, trend=trend)

    assert "時間趨勢不可用" in html
    assert "1970" in html
    assert "無法建立時序曲線" in html
    assert "<polyline" not in html
    assert "平均持倉      : 無法計算" in result.text_report
    assert "平均持倉      : 0.0 天" not in result.text_report
    assert "最大回撤      : 無法計算" in result.text_report
    assert not result.out_of_sample.available
    assert "樣本內: 2 筆" not in result.text_report


if __name__ == "__main__":
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
