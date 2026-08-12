import torch
import math

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.
    
    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    # Your code here
    if norm_type == "l1":
        return float(sum(abs(i) for i in arr))
    if norm_type == "l2":
        return math.sqrt(sum(i*i for i in arr))
    if norm_type == "frobenius":
        total = float(sum(i*i for i in arr[0]) + sum(i*i for i in arr[1]))
        return math.sqrt(total)
