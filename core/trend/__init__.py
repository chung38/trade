"""S3 時間趨勢分析:分期彙總、權益曲線、滾動期望值、優勢衰退偵測。"""

from .timeline import (
    DecayResult,
    EquityPoint,
    PeriodBucket,
    RollingPoint,
    TrendReport,
    analyze_trend,
    bucket_by_period,
    detect_decay,
    equity_curve,
    render_trend_text,
    rolling_expectancy,
)

__all__ = [
    "PeriodBucket",
    "EquityPoint",
    "RollingPoint",
    "DecayResult",
    "TrendReport",
    "analyze_trend",
    "bucket_by_period",
    "equity_curve",
    "rolling_expectancy",
    "detect_decay",
    "render_trend_text",
]
