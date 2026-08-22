import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

def main():
    # 1. Literature & Baseline
    # BiLSTM literature typically 15.61 for FD002
    results = {
        "Lit. BiLSTM": 15.61,
        "KePIN Baseline (raw3)": 14.5673,
    }
    
    # 2. Add Screening variants (50 epoch) 
    phase3 = "experiments_result/ablation_conditioning/phase3/results_table.csv"
    phase4 = "experiments_result/ablation_conditioning/phase4/results_table.csv"
    
    screening_results = {}
    for p in [phase3, phase4]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                screening_results[f"{row['variant']} (50ep)"] = float(row['rmse'])
                
    results.update(screening_results)
    
    # 3. Add Phase 5 Confirmation (250 epoch) if available
    phase5 = "experiments_result/ablation_conditioning/phase5_confirm/results_table.csv"
    if os.path.exists(phase5):
        df = pd.read_csv(phase5)
        for _, row in df.iterrows():
            if row['rmse'] != "ERROR":
                results[f"{row['variant']} (250ep)"] = float(row['rmse'])
                
    # Plotting
    labels = list(results.keys())
    values = list(results.values())
    
    # Sort for better visualization
    sorted_idx = np.argsort(values)
    labels = [labels[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, values, color='skyblue')
    
    # Highlight baselines and winners
    for i, label in enumerate(labels):
        if "Baseline" in label:
            bars[i].set_color('orange')
        elif "250ep" in label:
            bars[i].set_color('green')
        elif "Lit" in label:
            bars[i].set_color('gray')
            
    plt.xlabel('RMSE (Lower is better)')
    plt.title('FD002 Conditioned Koopman Ablation Results')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', ha='left', va='center', fontweight='bold')
                 
    plt.tight_layout()
    out_path = "experiments_result/ablation_conditioning/ablation_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
