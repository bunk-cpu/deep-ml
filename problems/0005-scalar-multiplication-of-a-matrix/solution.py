import torch

def scalar_multiply(matrix, scalar) -> torch.Tensor:
    """
    Multiply each element of a 2D matrix by a scalar using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of the same shape.
    """
    # Convert input to tensor
    m_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    output_list = []
    for i in range(len(m_t)):
        child_list = []
        for j in range(len(m_t[i])):
            child_list.append(m_t[i][j]*scalar)
        output_list.append(child_list)
    return torch.as_tensor(output_list)
