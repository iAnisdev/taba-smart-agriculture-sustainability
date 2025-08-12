"""Preprocess utilities (placeholder for any dataset balancing or on-disk ops).
Current pipeline applies bias mitigation at dataloader level (sampler or class weights).
Add any image cleaning/resizing scripts here if needed.
"""
import os
from PIL import Image


def preprocess_data(data_dir: str):
    """
    Example preprocess: verify images and remove corrupt ones.
    Extend with normalization, resizing, or augmentation as needed.
    """
    print(f"[PREPROCESS] Verifying images in {data_dir}...")
    bad = 0
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_dir, split)
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            for f in os.listdir(class_path):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                p = os.path.join(class_path, f)
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception:
                    bad += 1
                    try:
                        os.remove(p)
                        print(f"[PREPROCESS] Removed corrupt image: {p}")
                    except OSError:
                        print(f"[PREPROCESS] Failed to remove: {p}")
    print(f"[PREPROCESS] Done. Removed {bad} corrupt images.")
