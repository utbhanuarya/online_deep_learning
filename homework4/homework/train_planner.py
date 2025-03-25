"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

import argparse
import torch
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data

def main():

    # Parse command-line arguments for flexibility
    parser = argparse.ArgumentParser(description="Train driving waypoint planner (MLP, Transformer, or CNN).")
    parser.add_argument("--model", type=str, required=True, choices=["mlp_planner", "transformer_planner", "cnn_planner"],
                        help="Which planner model to train.")
    parser.add_argument("--train_dir", type=str, default="drive_data/train", 
                        help="Path to training dataset directory (containing episodes).")
    parser.add_argument("--val_dir", type=str, default="drive_data/val", 
                        help="Path to validation dataset directory (containing episodes).")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--no_cuda", action="store_true", help="Force training on CPU (ignore GPU even if available).")
    args = parser.parse_args()

    # Select the appropriate data transform pipeline based on model
    if args.model in ["mlp_planner", "transformer_planner"]:
        transform_pipeline = "state_only"   # only load track and waypoint data
    else:
        transform_pipeline = "default"      # load image + track data

    # Load training and validation data
    train_loader = load_data(args.train_dir, transform_pipeline=transform_pipeline, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    # Only load validation data if directory exists
    try:
        val_loader = load_data(args.val_dir, transform_pipeline=transform_pipeline, batch_size=args.batch_size, shuffle=False)
    except Exception as e:
        print(f"Warning: Validation data not loaded ({e}). Proceeding without validation.")

    # Instantiate the selected model
    if args.model == "mlp_planner":
        model = MLPPlanner()
    elif args.model == "transformer_planner":
        model = TransformerPlanner()
    elif args.model == "cnn_planner":
        model = CNNPlanner()

    # Use GPU if available (unless disabled)
    # use_cuda = torch.cuda.is_available() and not args.no_cuda
    # device = torch.device("cuda" if use_cuda else "cpu")
    if torch.backends.mps.is_available() and not args.no_cuda:
        device = torch.device("mps")  # Use Apple MPS on MacBook
    elif torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")  # Use CUDA on supported devices
    else:
        device = torch.device("cpu")  # Fallback to CPU
    model.to(device)

    # Loss function (mean squared error for regression)
    criterion = torch.nn.MSELoss(reduction="none")
    # Optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_metric = PlannerMetric()  # track train errors
        for batch in train_loader:
            # Move data to device
            if args.model == "cnn_planner":
                images = batch["image"].to(device, dtype=torch.float32)
                preds = model(image=images)
            else:
                left = batch["track_left"].to(device, dtype=torch.float32)
                right = batch["track_right"].to(device, dtype=torch.float32)
                preds = model(track_left=left, track_right=right)
            # Ground truth waypoints and mask
            labels = batch["waypoints"].to(device, dtype=torch.float32)
            labels_mask = batch["waypoints_mask"].to(device)  # bool tensor
            # Compute MSE loss, masking out invalid waypoints
            loss_per_element = criterion(preds, labels)                 # shape (B, n_waypoints, 2)
            mask_expanded = labels_mask.unsqueeze(-1).to(dtype=torch.float32)  # (B, n_waypoints, 1)
            masked_loss = loss_per_element * mask_expanded              # zero-out invalid waypoint losses
            # Average loss over *valid* waypoint coordinates
            valid_coords = labels_mask.sum() * 2  # each valid waypoint contributes 2 coordinates
            if valid_coords.item() == 0:
                continue  # skip if no valid points (edge case)
            loss = masked_loss.sum() / valid_coords
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate training metrics (uses L1 errors internally)
            train_metric.add(preds, labels, labels_mask)
        # Compute average training errors for this epoch
        train_results = train_metric.compute()
        log_msg = (f"Epoch {epoch}/{args.epochs} - Training L1: {train_results['l1_error']:.4f}, "
                f"Longitudinal: {train_results['longitudinal_error']:.4f}, Lateral: {train_results['lateral_error']:.4f}")
        # Validation metrics
        if val_loader is not None:
            model.eval()
            val_metric = PlannerMetric()
            with torch.no_grad():
                for batch in val_loader:
                    if args.model == "cnn_planner":
                        images = batch["image"].to(device, dtype=torch.float32)
                        preds = model(image=images)
                    else:
                        left = batch["track_left"].to(device, dtype=torch.float32)
                        right = batch["track_right"].to(device, dtype=torch.float32)
                        preds = model(track_left=left, track_right=right)
                    labels = batch["waypoints"].to(device, dtype=torch.float32)
                    labels_mask = batch["waypoints_mask"].to(device)
                    val_metric.add(preds, labels, labels_mask)
            val_results = val_metric.compute()
            log_msg += (f" | Validation L1: {val_results['l1_error']:.4f}, Longitudinal: "
                        f"{val_results['longitudinal_error']:.4f}, Lateral: {val_results['lateral_error']:.4f}")
        print(log_msg)

    # Save the trained model weights
    model_path = save_model(model)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()

# # Train the MLP planner for 20 epochs
# python -m homework.train_planner --model mlp_planner --epochs 20 --batch_size 64 --lr 1e-3

# # Train the Transformer planner (may require more epochs or tuning)
# python -m homework.train_planner --model transformer_planner --epochs 30 --lr 5e-4

# # Train the CNN planner on images
# python -m homework.train_planner --model cnn_planner --epochs 20
