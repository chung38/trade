# -*- coding: utf-8 -*-
"""校準置中 bootstrap p 值的型一誤差(docs/methodology.md 引用)。

在「真實期望 = 0」的世界裡(零期望常態損益),單尾 α=0.05 的檢定
應該只有約 5% 的實驗被誤判為「顯著為正」。若明顯偏離,代表 p 值失準。

執行:PYTHONUTF8=1 python experiments/exp_bootstrap_calibration.py
固定 seed;約需 1~2 分鐘(400 次實驗 × 400 次重抽)。純標準庫。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.verdict.statistics import test_expectancy_positive  # noqa: E402

SEED = 42
TRIALS = 400
N = 40


def main() -> None:
    rng = random.Random(SEED)
    hits_boot = 0
    hits_t = 0
    for i in range(TRIALS):
        data = [rng.gauss(0, 100) for _ in range(N)]
        r = test_expectancy_positive(data, n_bootstrap=400, seed=i)
        if r.p_value_bootstrap < 0.05 and r.mean > 0:
            hits_boot += 1
        if r.p_value_t < 0.05 and r.mean > 0:
            hits_t += 1
    se = (0.05 * 0.95 / TRIALS) ** 0.5
    print(f"seed={SEED}, trials={TRIALS}, n={N}, 零期望常態損益")
    print(f"bootstrap(置中)型一誤差:{hits_boot / TRIALS:.1%}(理論 5%,SE≈{se:.1%})")
    print(f"t 檢定       型一誤差:{hits_t / TRIALS:.1%}(理論 5%)")


if __name__ == "__main__":
    main()
