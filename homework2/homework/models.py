"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        # raise NotImplementedError("ClassificationLoss.forward() is not implemented")
        return F.cross_entropy(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        # Calculate the size of the flattened input
        self.flattened_size = 3 * h * w  # 3 channels (RGB), height h, width w

        # Define the linear layer
        self.linear = nn.Linear(self.flattened_size, num_classes)
        # raise NotImplementedError("LinearClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor from (B, 3, H, W) to (B, 3 * H * W)
        x = x.view(x.size(0), -1)  # (B, 3 * H * W)

        # Pass through the linear layer
        logits = self.linear(x)  # (B, num_classes)

        return logits
        # raise NotImplementedError("LinearClassifier.forward() is not implemented")


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_size: int = 128,  # Size of the hidden layer
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        # Calculate the size of the flattened input
        self.flattened_size = 3 * h * w  # 3 channels (RGB), height h, width w

        # Define the MLP using nn.Sequential
        self.mlp = nn.Sequential(
            # Input layer (flattened input to hidden layer)
            nn.Linear(self.flattened_size, hidden_size),
            nn.ReLU(),  # Non-linear activation function
            # Output layer (hidden layer to num_classes)
            nn.Linear(hidden_size, num_classes),
        )
        # raise NotImplementedError("MLPClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor from (B, 3, H, W) to (B, 3 * H * W)
        x = x.view(x.size(0), -1)  # (B, 3 * H * W)

        # Pass through the MLP
        logits = self.mlp(x)  # (B, num_classes)

        return logits
        # raise NotImplementedError("MLPClassifier.forward() is not implemented")


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,  # Size of hidden layers
        num_layers: int = 4,    # Number of hidden layers
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        # Calculate the size of the flattened input
        self.flattened_size = 3 * h * w  # 3 channels (RGB), height h, width w

        # Define the deep MLP using nn.Sequential
        layers = []
        input_size = self.flattened_size

        # Add hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())  # Non-linear activation function
            input_size = hidden_dim  # Output of current layer is input to next

        # Add the final output layer
        layers.append(nn.Linear(hidden_dim, num_classes))

        # Combine all layers into a sequential container
        self.mlp = nn.Sequential(*layers)
        # raise NotImplementedError("MLPClassifierDeep.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor from (B, 3, H, W) to (B, 3 * H * W)
        x = x.view(x.size(0), -1)  # (B, 3 * H * W)

        # Pass through the deep MLP
        logits = self.mlp(x)  # (B, num_classes)

        return logits
        # raise NotImplementedError("MLPClassifierDeep.forward() is not implemented")


class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,  # Size of hidden layers
        num_layers: int = 5,    # Number of hidden layers (at least 4)
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        # Calculate the size of the flattened input
        self.flattened_size = 3 * h * w  # 3 channels (RGB), height h, width w

        # Define the input layer
        self.input_layer = nn.Linear(self.flattened_size, hidden_dim)

        # Define hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):  # Subtract 1 for the input layer
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)

        # Activation function
        self.relu = nn.ReLU()
        # raise NotImplementedError("MLPClassifierDeepResidual.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor from (B, 3, H, W) to (B, 3 * H * W)
        x = x.view(x.size(0), -1)  # (B, 3 * H * W)

        # Pass through the input layer
        x = self.input_layer(x)
        x = self.relu(x)

        # Pass through hidden layers with residual connections
        for layer in self.hidden_layers:
            residual = x  # Save the input for the residual connection
            x = layer(x)
            x = self.relu(x)
            x = x + residual  # Add the residual connection

        # Pass through the output layer
        logits = self.output_layer(x)  # (B, num_classes)

        return logits
        # raise NotImplementedError("MLPClassifierDeepResidual.forward() is not implemented")


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
