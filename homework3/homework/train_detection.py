if __name__ == '__main__':
    import torch
    from models import Detector, save_model
    from metrics import DetectionMetric
    from datasets.road_dataset import load_data

    # Configuration
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    num_epochs = 5
    batch_size = 32
    lr = 1e-3

    # Load training and validation data
    train_loader = load_data("drive_data/val", batch_size=batch_size, shuffle=True)
    val_loader   = load_data("drive_data/val", batch_size=batch_size, shuffle=False)

    # Initialize model, losses, optimizer, metric
    model = Detector().to(device)
    seg_criterion = torch.nn.CrossEntropyLoss()  # for segmentation (logits vs class indices)
    depth_criterion = torch.nn.L1Loss()          # for depth regression (absolute error)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metric = DetectionMetric(num_classes=3)

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)            # shape (B,3,96,128)
            depth_gt = batch["depth"].to(device)          # shape (B,96,128)
            track_gt = batch["track"].to(device)          # shape (B,96,128), int64
            # Forward pass
            logits, depth_pred = model(images)            # logits: (B,3,96,128), depth_pred: (B,96,128)
            # Compute losses
            seg_loss = seg_criterion(logits, track_gt)    # segmentation cross-entropy
            depth_loss = depth_criterion(depth_pred, depth_gt)  # L1 depth error
            loss = seg_loss + depth_loss                  # combined loss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                depth_gt = batch["depth"].to(device)
                track_gt = batch["track"].to(device)
                logits, depth_pred = model(images)
                # calculate losses for reporting
                seg_loss = seg_criterion(logits, track_gt)
                depth_loss = depth_criterion(depth_pred, depth_gt)
                val_loss += (seg_loss.item() + depth_loss.item()) * images.size(0)
                # Update detection metrics
                preds = logits.argmax(dim=1)             # (B,96,128) predicted class indices
                metric.add(preds.cpu(), track_gt.cpu(),
                           depth_pred.cpu(), depth_gt.cpu())
        val_loss = val_loss / len(val_loader.dataset)
        metrics = metric.compute()  # {'iou': ..., 'accuracy': ..., 'abs_depth_error': ..., 'tp_depth_error': ...}

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | "
              f"Val mIoU = {metrics['iou']:.3f}, Val Accuracy = {metrics['accuracy']:.3f}, "
              f"Depth MAE = {metrics['abs_depth_error']:.4f}, Lane MAE = {metrics['tp_depth_error']:.4f}")

    # Save trained detector model
    save_model(model)
