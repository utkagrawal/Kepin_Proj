import numpy as np
import warnings

def compute_local_linear_residual(window_data: np.ndarray) -> np.ndarray:
    """
    Fits an independent AR(1) linear model for each channel in the window 
    (predicting timestep t from timestep t-1), and computes the mean squared residual.
    
    Args:
        window_data: (seq_len, F) array where seq_len is the window length 
                     and F is the number of channels/sensors.
                     
    Returns:
        aggregated_residual: (1,) vector containing the mean across all channels 
                             of the channel's Mean Squared Residual.
    """
    seq_len, num_features = window_data.shape
    
    # We need at least 2 time steps to fit a t -> t+1 model
    if seq_len < 2:
        return np.array([0.0], dtype=np.float32)
        
    # X is t-1, Y is t
    X = window_data[:-1, :]  # (seq_len - 1, F)
    Y = window_data[1:, :]   # (seq_len - 1, F)
    
    # Calculate means
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    
    # Center the data
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    
    # Calculate variances and covariances
    # We add a small epsilon to the variance to avoid division by zero for constant channels
    X_var = np.var(X, axis=0)
    cov = np.mean(X_centered * Y_centered, axis=0)
    
    # Calculate slope (m) and intercept (c)
    m = cov / (X_var + 1e-8)
    c = Y_mean - m * X_mean
    
    # Predictions
    Y_pred = m * X + c
    
    # Residuals
    residuals = Y - Y_pred
    
    # Mean Squared Residual per channel. Shape: (F,)
    msr_per_channel = np.mean(residuals**2, axis=0)
    
    # Aggregate across channels by taking the mean. Shape: ()
    agg_msr = np.nanmean(msr_per_channel)
    
    # Return a 1-dimensional feature vector
    return np.array([agg_msr], dtype=np.float32)

if __name__ == "__main__":
    np.random.seed(42)
    # Test on a window that is nearly perfectly linear (low residual)
    t = np.arange(30)[:, None]
    linear_window = 2.5 * t + 1.0 + np.random.randn(30, 16) * 0.01  
    
    # Test on a highly non-linear/irregular window (high residual)
    nonlinear_window = np.sin(t) + np.random.randn(30, 16) * 2.0   
    
    feat_linear = compute_local_linear_residual(linear_window)
    feat_nonlinear = compute_local_linear_residual(nonlinear_window)
    
    print(f"Extracted features dimensionality: {feat_linear.shape[0]} dimensions")
    print(f"Linear Window MSR:     {feat_linear[0]:.4f}")
    print(f"Non-linear Window MSR: {feat_nonlinear[0]:.4f}")
