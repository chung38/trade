# -*- coding: utf-8 -*-
"""重現 survivorship 的精確機率,並用蒙地卡羅交叉驗證。

兩個「不同事件」(docs/faq.md 與 SKILL.md 引用的數字):
  A. 1000 人各猜 10 次,「至少一人 10 全對」   (p=0.5)
  B. 1000 人各猜 20 次,「至少一人出現 10 連勝」(CLI 預設 --trials 20)

執行:PYTHONUTF8=1 python experiments/exp_survivorship.py
純標準庫;蒙地卡羅部分固定 seed。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.survivorship import prob_at_least_one_streak  # noqa: E402

SEED = 20260713
N_MC = 20000


def mc_at_least_one_streak(n_traders: int, n_trials: int, streak: int) -> float:
    rng = random.Random(SEED)
    hits = 0
    for _ in range(N_MC):
        found = False
        for _ in range(n_traders):
            run = 0
            for _ in range(n_trials):
                if rng.random() < 0.5:
                    run += 1
                    if run >= streak:
                        found = True
                        break
                else:
                    run = 0
            if found:
                break
        hits += found
    return hits / N_MC


def main() -> None:
    for label, trials in (("A(每人 10 次,10 全對)", 10), ("B(每人 20 次,10 連勝)", 20)):
        exact = prob_at_least_one_streak(n_traders=1000, n_trials=trials, streak=10)
        approx = mc_at_least_one_streak(50, trials, 10)  # MC 用 50 人縮小規模驗證公式
        exact_50 = prob_at_least_one_streak(n_traders=50, n_trials=trials, streak=10)
        print(f"{label}: 精確解 {exact:.1%}(1000 人)|"
              f" 交叉驗證(50 人): 精確 {exact_50:.3%} vs 蒙地卡羅 {approx:.3%}"
              f"(N={N_MC}, seed={SEED})")


if __name__ == "__main__":
    main()
