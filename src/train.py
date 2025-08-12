
from __future__ import annotations
import os
import torch
from torchvision import datasets, transforms
import timm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
import yaml
import json
from sklearn.metrics import confusion_matrix


def train_model(model_name, data_dir, augment=None, config=None):
    # Determine epochs from config or default to 12
    epochs = 12
    if config and isinstance(config, dict) and 'epochs' in config:
        epochs = int(config['epochs'])
    elif config and isinstance(config, str):
        try:
            with open(config) as f:
                cfg = yaml.safe_load(f)
            if 'epochs' in cfg:
                epochs = int(cfg['epochs'])
        except Exception:
            pass
    print(
        f"Training {model_name} on data in {data_dir} with augment={augment} and config={config}"
    )
    # Minimal real training loop for CIFAR-10
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dir = os.path.join(data_dir, "train")
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    num_classes = len(train_ds.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Progress bar is imported at top

    # Prepare weights directory
    weights_dir = os.path.join("weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{model_name}.pt")
    metrics_path = os.path.join(weights_dir, f"{model_name}.json")

    # Track metrics
    epoch_metrics = []
    model.train()
    for epoch in range(epochs):
        total, correct, running_loss = 0, 0, 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            running_loss += loss.item() * y.size(0)
        acc = correct / total
        avg_loss = running_loss / total
        print(f"[TRAIN] Epoch {epoch+1}: train_acc={acc:.4f}, avg_loss={avg_loss:.4f}")
        epoch_metrics.append({
            "epoch": epoch+1,
            "train_acc": acc,
            "train_loss": avg_loss
        })

    # Save model weights
    torch.save(model.state_dict(), weights_path)
    print(f"[TRAIN] Saved weights to {weights_path}")

    # Compute confusion matrix (imported at top)
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    cm = confusion_matrix(all_targets, all_preds)

    # Save metrics and metadata (imported at top)
    meta = {
        "model": model_name,
        "epochs": epochs,
        "metrics": epoch_metrics,
        "confusion_matrix": cm.tolist(),
        "img_size": 224,
        "batch_size": train_loader.batch_size,
        "optimizer": "Adam",
        "pretrained": True
    }
    with open(metrics_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[TRAIN] Saved metrics to {metrics_path}")

    print(f"[TRAIN] Done training {model_name}.")
    return {"status": "success", "model": model_name, "train_acc": epoch_metrics[-1]["train_acc"], "epochs": epochs, "weights": weights_path, "metrics": metrics_path}
