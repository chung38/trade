# -*- coding: utf-8 -*-
"""重現 docs/methodology.md 的多重比較實驗。

問題:對 K 個策略標籤各做一次顯著性檢定、不做多重比較校正,
「至少一個純隨機標籤被誤判為具優勢」的機率是多少?

方法:每個標籤 = 30 筆零期望常態損益(mean=0, sd=100),
對每個標籤做單尾 t 檢定(alpha=0.05),重複 N_SIM 次,
統計「至少一個標籤 p<0.05 且均值為正」的比例。

執行:PYTHONUTF8=1 python experiments/exp_multiple_comparison.py
固定 seed,結果可完全重現。純標準庫。
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.verdict.statistics import _student_t_sf  # noqa: E402

SEED = 20260713
N_SIM = 2000
N_TRADES = 30
ALPHA = 0.05


def one_tag_significant(rng: random.Random) -> bool:
    pnls = [rng.gauss(0, 100) for _ in range(N_TRADES)]
    mean = statistics.fmean(pnls)
    if mean <= 0:
        return False
    sd = statistics.stdev(pnls)
    if sd == 0:
        return False
    t = mean / (sd / N_TRADES ** 0.5)
    return _student_t_sf(t, N_TRADES - 1) < ALPHA


def main() -> None:
    print(f"seed={SEED}, n_sim={N_SIM}, 每標籤 {N_TRADES} 筆零期望損益, α={ALPHA}")
    print("| 策略標籤數 K | 至少一個被誤判為「具優勢」 |")
    print("|---|---|")
    for k in (1, 5, 10):
        rng = random.Random(SEED + k)
        hits = 0
        for _ in range(N_SIM):
            if any(one_tag_significant(rng) for _ in range(k)):
                hits += 1
        print(f"| {k} | {hits / N_SIM:.1%} |")


if __name__ == "__main__":
    main()
