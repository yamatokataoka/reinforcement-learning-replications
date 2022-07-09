from typing import List, Type

from torch import Tensor, nn


class MLP(nn.Module):
    """
    The multilayer perceptron (MLP)

    :param sizes: (List[int]) The dimensions of all layer(s).
    :param activation_function: (Type[nn.Module]) The activation function.
    :param output_activation_function: (Type[nn.Module]) The output activation function.
    """

    def __init__(
        self,
        sizes: List[int],
        activation_function: Type[nn.Module] = nn.Tanh,
        output_activation_function: Type[nn.Module] = nn.Identity,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            if i < len(sizes) - 2:
                current_activation_function = activation_function
            else:
                current_activation_function = output_activation_function
            layers += [nn.Linear(sizes[i], sizes[i + 1]), current_activation_function()]

        self.network: nn.Module = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass in the MLP

        :param input: (Tensor) The input tensor.
        :return output: (Tensor) The output tensor.
        """
        output: Tensor = self.network(input)
        return output
