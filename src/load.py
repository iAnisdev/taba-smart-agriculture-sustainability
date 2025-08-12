
from __future__ import annotations
import os
import torch
from torchvision import datasets
from torchvision.utils import save_image

def download_cifar10(data_dir: str):
    print(f"[INFO] Creating {data_dir} and subfolders if missing...")
    os.makedirs(data_dir, exist_ok=True)
    for split, train_flag in zip(["train", "val", "test"], [True, False, False]):
        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"[INFO] Downloading CIFAR-10 for split: {split}")
        ds = datasets.CIFAR10(root="data/cifar10", train=train_flag, download=True)
        # For val, use half of test set
        if split == "val":
            ds = torch.utils.data.Subset(ds, range(0, len(ds)//2))
        elif split == "test":
            ds = torch.utils.data.Subset(ds, range(len(ds)//2, len(ds)))
        from torchvision import transforms
        for idx in range(len(ds)):
            img, label = ds[idx]
            class_dir = os.path.join(split_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img_tensor = transforms.ToTensor()(img)
            save_image(img_tensor, os.path.join(class_dir, f"{idx}.png"))
        print(f"[INFO] Finished split: {split}, images saved.")
    print(f"[INFO] CIFAR-10 download and conversion complete.")
    # Cleanup: remove .tar.gz and extracted batches
    cifar_root = "data/cifar10"
    tar_path = os.path.join(cifar_root, "cifar-10-python.tar.gz")
    batches_path = os.path.join(cifar_root, "cifar-10-batches-py")
    try:
        if os.path.exists(tar_path):
            os.remove(tar_path)
            print(f"[INFO] Removed {tar_path}")
        if os.path.exists(batches_path):
            import shutil
            shutil.rmtree(batches_path)
            print(f"[INFO] Removed {batches_path}")
    except Exception as e:
        print(f"[WARN] Cleanup failed: {e}")

def summarize(data_dir: str):
    print(f"[INFO] Summary for {data_dir}:")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  {split}: MISSING")
            continue
        class_counts = {}
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.endswith('.png')])
                class_counts[class_name] = count
        print(f"  {split}: {sum(class_counts.values())} images, classes: {class_counts}")
