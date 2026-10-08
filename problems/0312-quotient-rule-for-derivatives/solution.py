import torch

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> torch.Tensor:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x) as a scalar torch.Tensor
    """
    # Your code here
    g = 0
    for i in range(len(g_coeffs)):
        g += g_coeffs[i] * (x**(len(g_coeffs) - i - 1))
    h = 0
    for i in range(len(h_coeffs)):
        h += h_coeffs[i] * (x**(len(h_coeffs) - i - 1))
    dtg = 0
    for i in range(len(g_coeffs)-1):
        dtg += g_coeffs[i] * (len(g_coeffs) - i - 1) * (x**(len(g_coeffs) - i - 2))
    dth = 0
    for i in range(len(h_coeffs)-1):
        dth += h_coeffs[i] * (len(h_coeffs) - i - 1) * (x**(len(h_coeffs) - i - 2))
    f = (dtg*h - g*dth)/(h**2)
    return torch.tensor(f, dtype=float)