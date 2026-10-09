import torch

def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll using PyTorch.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    # Your code here
    expected_value = (1 + n)/2
    total = 0
    for i in range(n):
        var = (i+1-expected_value)**2
        total += var
    variance = total / n
    return (expected_value, variance)