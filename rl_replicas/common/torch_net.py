from typing import List, Type

import torch
import torch.nn as nn

def mlp(
  sizes: List[int],
  activation: Type[nn.Module] = nn.Tanh,
  output_activation: Type[nn.Module] = nn.Identity
) -> nn.Module:
  layers: List[nn.Module] = []

  for j in range(len(sizes)-1):
    current_activation = activation if j < len(sizes)-2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j+1]), current_activation()]

  return nn.Sequential(*layers)
