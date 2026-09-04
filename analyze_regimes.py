#!/usr/bin/env python3
"""
Regime Analysis and Visualization for KePIN.
Reads saved predictions and regime sequences (pi_seq) to generate plots.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_regime_probabilities(y_true, pi_seq, out_path, num_samples=1000):
    """
    Plot regime probabilities (pi) over time (true RUL) for a contiguous trajectory.
    Assuming the test set preserves engine sequence order, we can plot a subset.
    """
    if pi_seq is None or len(pi_seq) == 0:
        return
        
    # Take a chunk that corresponds to one or more full engine cycles
    # For CMAPSS, true RUL goes down to 0 at the end of an engine's life.
    y_subset = y_true[:num_samples].flatten()
    pi_subset = pi_seq[:num_samples] # (num_samples, T, R)
    
    # We plot the probability at the last timestep of each sequence
    pi_last = pi_subset[:, -1, :] # (num_samples, R)
    
    # Find endpoints where RUL goes up (new engine)
    diffs = np.diff(y_subset)
    engine_starts = np.where(diffs > 10)[0] + 1
    engine_starts = [0] + list(engine_starts) + [num_samples]
    
    # Plot the first 3 engines
    num_engines = min(3, len(engine_starts) - 1)
    
    fig, axes = plt.subplots(num_engines, 1, figsize=(10, 4 * num_engines), sharey=True)
    if num_engines == 1:
        axes = [axes]
        
    for i in range(num_engines):
        start = engine_starts[i]
        end = engine_starts[i+1]
        
        y_eng = y_subset[start:end]
        pi_eng = pi_last[start:end]
        
        ax = axes[i]
        
        # Plot each regime's probability
        R = pi_eng.shape[1]
        for r in range(R):
            ax.plot(y_eng, pi_eng[:, r], label=f'Regime {r+1}', alpha=0.8, linewidth=2)
            
        ax.set_title(f"Test Engine {i+1} Regime Evolution")
        ax.set_xlabel("True RUL (Cycles Remaining)")
        ax.set_ylabel("Regime Probability (π)")
        ax.invert_xaxis() # RUL goes from high to low
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load predictions
    pred_path = os.path.join(args.res_dir, "predictions.npz")
    if not os.path.exists(pred_path):
        print(f"Predictions not found at {pred_path}")
        return
        
    preds = np.load(pred_path)
    y_true = preds['y_true']
    
    # Load pi_seq
    pi_path = os.path.join(args.res_dir, "pi_seq.npy")
    if os.path.exists(pi_path):
        pi_seq = np.load(pi_path)
        plot_path = os.path.join(args.out_dir, "regime_evolution.png")
        plot_regime_probabilities(y_true, pi_seq, plot_path)
        print(f"Saved regime plot to {plot_path}")
        
        # Calculate stats
        pi_last = pi_seq[:, -1, :]
        avg_pi = pi_last.mean(axis=0)
        entropy = -(pi_last * np.log(pi_last + 1e-10)).sum(axis=1).mean()
        
        print("\nRegime Statistics:")
        print(f"Average probability per regime: {np.round(avg_pi, 3)}")
        print(f"Mean Regime Entropy: {entropy:.3f} (Max for {pi_last.shape[1]} regimes: {np.log(pi_last.shape[1]):.3f})")
    else:
        print("No pi_seq.npy found (model might not use regimes).")

if __name__ == "__main__":
    main()
