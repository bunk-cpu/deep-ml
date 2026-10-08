import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
    res_list = []
    for _a in a_t:
        child_list = []
        for j in range(b_t.size(1)):
            b_mult = [_b[j] for _b in b_t]
            res = 0
            for k in range(len(b_mult)):
                res += _a[k] * b_mult[k]
            child_list.append(res)
        res_list.append(child_list)
    return torch.tensor(res_list)
