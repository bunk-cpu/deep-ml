import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    m, n = a_t.size()
    res = []
    if mode == 'row':
        for i in range(m):
            row = a_t[i]
            total_row_tensor = row.sum()
            mean_row = total_row_tensor.item()/n
            res.append(mean_row)
    if mode == 'column':
        for i in range(n):
            total_column = 0
            for j in range(m):
                total_column += a_t[j, i].item()
            mean_column = total_column / m
            res.append(mean_column)
    return torch.tensor(res, dtype=torch.float32)
