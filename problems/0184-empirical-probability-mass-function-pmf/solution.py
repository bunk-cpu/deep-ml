import torch

def empirical_pmf(samples: torch.Tensor) -> list:
    """
    Given a 1D tensor of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    PMF_list = []
    values, counts = torch.unique(samples, return_counts=True)
    for i in range(values.size(0)):
        target = values[i].item()
        target_count = counts[i].item()
        probability = target_count / samples.size(0)
        PMF_list.append((target, probability))
    
    return PMF_list
