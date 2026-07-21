import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math

def apply_rope(x, cos, sin):
	half = x.shape[-1] // 2
	first_half = x[..., :half]
	second_half = x[..., half:]
	new_first_half = first_half * cos - second_half * sin
	new_second_half = first_half * sin + second_half * sin
	out = torch.cat([new_first_half, new_second_half], dim = -1)
	return out
