import numpy as np
import warnings
from scipy.stats import skew, kurtosis

def compute_window_moments(window_data: np.ndarray) -> np.ndarray:
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    """
    Computes statistical moments over a temporal window for each channel, 
    and then aggregates across channels.
    
    Args:
        window_data: (seq_len, F) array where seq_len is the window length 
                     (e.g., 30) and F is the number of channels/sensors.
                     
    Returns:
        aggregated_features: (4,) vector containing:
            1. Mean across channels of the per-channel temporal means
            2. Mean across channels of the per-channel temporal variances
            3. Mean across channels of the per-channel temporal skewnesses
            4. Mean across channels of the per-channel temporal kurtoses
    """
    # 1. Compute per-channel temporal moments over the time dimension (axis=0)
    # Shape of each will be (F,)
    ch_mean = np.mean(window_data, axis=0)
    ch_var = np.var(window_data, axis=0)
    
    # Use bias=False for sample statistics and omit NaNs if any exist
    ch_skew = skew(window_data, axis=0, bias=False, nan_policy='omit')
    ch_kurt = kurtosis(window_data, axis=0, bias=False, nan_policy='omit')
    
    # 2. Aggregate across all F channels by taking the mean.
    # Using np.nanmean to safely ignore NaNs (e.g., from constant channels where skew/kurtosis is undefined)
    agg_mean = np.nanmean(ch_mean)
    agg_var = np.nanmean(ch_var)
    agg_skew = np.nanmean(ch_skew)
    agg_kurt = np.nanmean(ch_kurt)
    
    # Return exactly a 4-dimensional feature vector
    return np.array([agg_mean, agg_var, agg_skew, agg_kurt], dtype=np.float32)

if __name__ == "__main__":
    # Test on a dummy window (e.g., 30 time steps, 16 features/channels)
    np.random.seed(42)
    dummy_window = np.random.randn(30, 16)
    
    # Introduce a constant channel to verify NaN handling for skew/kurtosis
    dummy_window[:, 0] = 5.0 
    
    features = compute_window_moments(dummy_window)
    print(f"Extracted features dimensionality: {features.shape[0]} dimensions")
    print(f"Feature Vector: {features}")
    print("\nFeature Map:")
    print(f" 1. Aggregated Mean     : {features[0]:.4f}")
    print(f" 2. Aggregated Variance : {features[1]:.4f}")
    print(f" 3. Aggregated Skewness : {features[2]:.4f}")
    print(f" 4. Aggregated Kurtosis : {features[3]:.4f}")
