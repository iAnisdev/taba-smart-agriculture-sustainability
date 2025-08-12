
import os
import argparse
import json
import yaml
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(data_dir, model_name, config=None):
    """
    Basic evaluation for CIFAR-10. Loads model and computes accuracy on test set.
    """
    print(f"[EVAL] Evaluating {model_name} on {data_dir}")
    test_t = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_t)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    num_classes = len(test_ds.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    weights_path = os.path.join("weights", f"{model_name}.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[EVAL] Loaded weights from {weights_path}")
    else:
        print(f"[EVAL] No weights found at {weights_path}, evaluating random model.")
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(list(preds))
            y_true.extend(list(y.cpu().numpy()))
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"[EVAL] Test accuracy: {acc:.4f}")
    print("[EVAL] Classification report:")
    print(classification_report(y_true, y_pred))
    print("[EVAL] Confusion matrix:")
    print(cm)
    # Save results to JSON
    results = {
        "model": model_name,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "num_classes": num_classes,
        "test_size": len(test_ds)
    }
    metrics_path = os.path.join("weights", f"{model_name}_eval.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EVAL] Saved evaluation metrics to {metrics_path}")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, required=True, help='runs/<timestamp>_<model>')
    p.add_argument('--config', type=str, default='src/config.yaml')
    return p.parse_args()


def main():

    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Infer model name from run_dir
    model_name = os.path.basename(args.run_dir).split('_', 2)[-1]
    data_dir = cfg['paths']['data_dir'] if 'paths' in cfg and 'data_dir' in cfg['paths'] else 'data/cifar10'
    evaluate_model(data_dir, model_name, config=args.config)



if __name__ == "__main__":
    main()
