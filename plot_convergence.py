import pandas as pd
import matplotlib.pyplot as plt
import os

artifact_dir = "/userhome/mtech/a.utkarsh/.gemini/antigravity-ide/brain/1d643d03-bcaf-42c8-8f96-adeed32f2103"
res_dir = "experiments_result_half1"

def plot_dataset(dataset_name):
    csv_path = os.path.join(res_dir, f"history_CMAPSS_{dataset_name}_run0.csv")
    if not os.path.exists(csv_path):
        return
    
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 5))
    
    # Plot Train Loss
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
    plt.title(f'{dataset_name} - Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot Val RMSE
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val_rmse'], label='Val RMSE', color='red')
    plt.title(f'{dataset_name} - Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(artifact_dir, f"{dataset_name}_convergence.png"))
    plt.close()

plot_dataset("FD001")
plot_dataset("FD003")
