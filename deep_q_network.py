import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HEIGHT = 84
WIDTH = 84

DIR_MODEL = "models/"


class DeepQNetwork(nn.Module):
    def __init__(self, input_channels: int, num_actions: int):
        """
        Initializes the neural network.

        Args:
        - input_channels (int): Number of input channels (default is 4 for 84x84x4 input image).
        - num_actions (int): Number of valid actions (output size).
        """
        super(DeepQNetwork, self).__init__()

        for id_model in range(int(1e5)):
            path_model = self._get_path_model(id_model)
            if not os.path.exists(path_model):
                self.id_model = id_model
                break

        # First convolutional layer: 16 filters of size 8x8, stride 4
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)

        # Second convolutional layer: 32 filters of size 4x4, stride 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Fully connected hidden layer with 256 rectifier units
        self.fc1 = nn.Linear(32 * 9 * 9, 256)

        # Output layer: one output per valid action
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        """

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(data=x, dtype=torch.float)

        assert x.shape[1:] == (4, HEIGHT, WIDTH)

        # Apply first convolutional layer and ReLU
        x = F.relu(self.conv1(x))

        # Apply second convolutional layer and ReLU
        x = F.relu(self.conv2(x))

        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected hidden layer and ReLU
        x = F.relu(self.fc1(x))

        # Apply the output layer (linear layer)
        x = self.fc2(x)

        return x

    @staticmethod
    def _get_path_model(id_model: int) -> str:
        return os.path.join(DIR_MODEL, f"deepqnetwork_weights-{id_model}.pth")

    @classmethod
    def load(cls, id_model: int) -> "DeepQNetwork":
        model = DeepQNetwork(input_channels=4, num_actions=6)

        path = cls._get_path_model(id_model)
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

        return model

    def save(self) -> None:
        torch.save(self.state_dict(), self._get_path_model(self.id_model))


def _main():
    input_image = torch.randn(1, 4, HEIGHT, WIDTH)
    num_actions = 6

    # Create the network
    dqn = DeepQNetwork(input_channels=4, num_actions=num_actions)

    # Forward pass
    output = dqn(input_image)
    print("Output shape:", output)  # Should be (1, num_actions)


if __name__ == "__main__":
    _main()
