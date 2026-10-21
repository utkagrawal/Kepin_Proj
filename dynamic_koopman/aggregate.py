#!/usr/bin/env python3
"""Collect run results into per-arm tables (mean ± std over runs + 3-run ensemble).

Usage: python aggregate.py [runs_dir]
Writes <runs_dir>/summary.csv and prints one table per protocol.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

PAPER_KEPIN = {"FD001": 12.92, "FD002": 15.08, "FD003": 11.42, "FD004": 17.04}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    rows = []
    for arm_dir in sorted(glob.glob(os.path.join(root, "*", "*", "*"))):
        protocol, dataset, arm = arm_dir.split(os.sep)[-3:]
        res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(arm_dir, "run*_result.json")))]
        if not res:
            continue
        preds = [np.load(f) for f in sorted(glob.glob(os.path.join(arm_dir, "run*_predictions.npz")))]
        rmses = np.array([r["rmse"] for r in res])
        y_true = preds[0]["y_true"]
        ens = np.mean([p["y_pred"] for p in preds], axis=0)
        rows.append({
            "protocol": protocol, "dataset": dataset, "arm": arm, "n_runs": len(res),
            "rmse_mean": rmses.mean(), "rmse_std": rmses.std(ddof=1) if len(res) > 1 else np.nan,
            "rmse_runs": " ".join(f"{x:.2f}" for x in rmses),
            "rmse_ensemble": float(np.sqrt(np.mean((y_true - ens) ** 2))),
            "mae_mean": np.mean([r["mae"] for r in res]),
            "score_mean": np.mean([r["score"] for r in res]),
            "sec_per_epoch": np.mean([r["sec_per_epoch"] for r in res]),
            "epochs": np.mean([r["epochs_trained"] for r in res]),
            "paper_kepin": PAPER_KEPIN[dataset],
        })
    if not rows:
        print("no results yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    for protocol, g in df.groupby("protocol"):
        print(f"\n=== protocol: {protocol} ===")
        print(g.drop(columns="protocol").to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
