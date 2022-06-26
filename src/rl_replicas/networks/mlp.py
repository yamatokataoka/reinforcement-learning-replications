from torch import Tensor, nn


class MLP(nn.Module):
    """
    The multilayer perceptron (MLP)

    :param sizes: (list[int]) The network size
    :param activation_function: (type[nn.Module]) The activation function
    :param output_activation_function: (type[nn.Module]) The output activation function
    """

    def __init__(
        self,
        sizes: list[int],
        activation_function: type[nn.Module] = nn.Tanh,
        output_activation_function: type[nn.Module] = nn.Identity,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
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

        :param input: (Tensor) The input value
        :return output: (Tensor) The output value
        """
        output: Tensor = self.network(input)
        return output
