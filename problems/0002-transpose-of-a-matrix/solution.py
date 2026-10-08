import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.
    
    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A transposed torch.Tensor
    """
    a_t = torch.as_tensor(a)
    # Your code here
    output_list = []
    output_len = len(a_t[0])
    for i in range(output_len):
        child_list = []
        for j in range(len(a_t)):
            child_list.append(a_t[j][i])
        output_list.append(child_list)
    return torch.as_tensor(output_list)