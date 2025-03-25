from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        # Flattened input size: both boundaries (2) * n_track * 2 coordinates
        input_dim = 2 * n_track * 2  # 40 for default n_track=10
        hidden_dim = 128
        # Define a 3-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_waypoints * 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # raise NotImplementedError
        B = track_left.size(0)
        # Concatenate and flatten left/right track inputs
        x = torch.cat([track_left, track_right], dim=1).view(B, -1)  # shape (B, 40)
        # Forward pass through MLP
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)  # (B, n_waypoints*2)
        return out.view(B, self.n_waypoints, 2)

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Linear projection for input track points (2 -> d_model features)
        self.input_proj = nn.Linear(2, d_model)
        # Transformer decoder layers for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        # Output layer to map decoder output to 2D waypoint coordinates
        self.fc_out = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # raise NotImplementedError
        B = track_left.size(0)
        # Combine left and right track points and project to feature space
        # track_feat: (B, 2*n_track, d_model)
        track_points = torch.cat([track_left, track_right], dim=1)
        track_feat = self.input_proj(track_points)
        # Prepare queries: start with learned embeddings of shape (n_waypoints, d_model)
        # Expand to (B, n_waypoints, d_model) so each batch uses the same initial queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        # Cross-attention: each waypoint query attends to all track point features
        decoder_output = self.transformer_decoder(queries, track_feat)  # (B, n_waypoints, d_model)
        # Final linear layer to predict 2D coordinates for each waypoint query
        out = self.fc_out(decoder_output)  # (B, n_waypoints, 2)
        return out

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        # Convolutional layers with increasing channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)   # output size: 16×96×128
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # output size: 32×96×128
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # output size: 64×96×128
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2) # output size: 128×96×128
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling halves H and W
        # Fully connected regression layers
        self.fc1 = nn.Linear(128 * 6 * 8, 128)           # flatten 128@6x8 into 128-dim hidden
        self.fc2 = nn.Linear(128, n_waypoints * 2)       # output 6 values (3*2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # raise NotImplementedError
        # Apply conv layers with ReLU and pooling
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        # Flatten and apply fully-connected layers
        x = x.view(x.size(0), -1)        # flatten to (B, 6144)
        x = torch.relu(self.fc1(x))      # hidden layer
        out = self.fc2(x)               # (B, 6) output
        return out.view(x.size(0), self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
