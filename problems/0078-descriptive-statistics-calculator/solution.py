import torch
import math
def f(_tensor):
    median_idx = _tensor.size(0)/2
    if int(median_idx) == median_idx:
        median_data = (_tensor[int(median_idx)].item() + _tensor[int(median_idx) - 1].item()) / 2
    else:
        median_data = _tensor[int(median_idx)].item()
    return median_data

def descriptive_statistics(data) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset using PyTorch.
    
    Args:
        data: List, torch.Tensor, or array-like of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data, _ = torch.sort(torch.as_tensor(data, dtype=torch.float32))
    mean_data = sum(data.tolist())/data.size(0)
    
    median_data = f(data)
    
    values, counts = torch.unique(data, return_counts=True)
    max_idx = counts.argmax()
    mode_value = values[max_idx].item()

    variance = sum([(x.item() - mean_data)**2 for x in data]) / data.size(0)
    standard_deviation = math.sqrt(variance)

    median_idx = data.size(0)/2
    half_length = int(data.size(0) - median_idx)
    left_half_median_data = f(data[:half_length+1])
    right_half_median_data = f(data[half_length:])
    interquartile_range = right_half_median_data - left_half_median_data
    return {
        'mean':mean_data,
        'median':median_data,
        'mode':mode_value,
        'variance':variance,
        'standard_deviation':standard_deviation,
        '25th_percentile': left_half_median_data,
        '50th_percentile': median_data,
        '75th_percentile': right_half_median_data,
        'interquartile_range': interquartile_range
    }