import torch
from models import Classifier, save_model
from metrics import AccuracyMetric
from datasets.classification_dataset import load_data

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 128
learning_rate = 1e-3

# Data loaders for train and validation sets
train_loader = load_data("classification_data/train", transform_pipeline="aug",
                         batch_size=batch_size, shuffle=True)
val_loader   = load_data("classification_data/val", transform_pipeline="default",
                         batch_size=batch_size, shuffle=False)

# Initialize model, loss function, optimizer, metric
model = Classifier().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
metric = AccuracyMetric()

for epoch in range(num_epochs):
    model.train()
    metric.reset()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Forward pass and loss
        logits = model(images)                  # (B,6) logits
        loss = criterion(logits, labels)        # cross-entropy loss
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Accumulate training metrics
        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        metric.add(preds, labels)              # update correct/total&#8203;:contentReference[oaicite:34]{index=34}
    # Compute average training loss and accuracy
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = metric.compute()["accuracy"]    # training accuracy&#8203;:contentReference[oaicite:35]{index=35}

    # Validation loop
    model.eval()
    metric.reset()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            metric.add(preds, labels)
    val_loss /= len(val_loader.dataset)
    val_acc = metric.compute()["accuracy"]      # validation accuracy

    # Log epoch metrics
    print(f"Epoch {epoch+1:02d}: "
          f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f} | "
          f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

# Save the trained model weights
save_model(model)
