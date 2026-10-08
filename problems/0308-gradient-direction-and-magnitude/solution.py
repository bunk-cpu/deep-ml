import torch
import math

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	# Your code here
	magnitude = math.sqrt(sum(x**2 for x in gradient))
	if magnitude == 0:
		return {
			'magnitude': 0,
			'direction': torch.tensor(gradient, dtype=torch.float32),
			'descent_direction': torch.tensor(gradient, dtype=torch.float32)
		}
	direction = [i/magnitude for i in gradient]
	descent_direction = [-i/magnitude for i in gradient]
	return {
		'magnitude': magnitude,
		'direction': torch.tensor(direction, dtype=torch.float32),
		'descent_direction': torch.tensor(descent_direction, dtype=torch.float32)
	}