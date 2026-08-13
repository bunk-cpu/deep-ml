import torch
import torch.nn.functional as F

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    # Implement your code here
    dot_product = float(torch.matmul(v1, v2))
    norm_v1 = float(torch.sqrt(sum(i*i for i in v1)))
    norm_v2 = float(torch.sqrt(sum(i*i for i in v2)))
    cos = dot_product / (norm_v1*norm_v2)
    return cos