import argparse
import sys

from load import download_cifar10, summarize
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Unified CLI for Smart Agriculture Pipeline")
    parser.add_argument('--load-data', '--ld', action='store_true', help='Download CIFAR-10 dataset')
    parser.add_argument('--preprocess', '--pp', action='store_true', help='Preprocess and clean data')
    parser.add_argument('--train', '--tr', action='store_true', help='Train model')
    parser.add_argument('--evaluate', '--ev', action='store_true', help='Evaluate model')
    parser.add_argument('--all', '--a', action='store_true', help='Run the entire pipeline')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to train/evaluate')
    parser.add_argument('--config', '-c', type=str, help='Path to config file')
    args = parser.parse_args()

    DATA_DIR = "data/cifar10"

    if args.load_data or args.all:
        print("[CLI] Downloading CIFAR-10 dataset...")
        download_cifar10(DATA_DIR)
        summarize(DATA_DIR)

    if args.preprocess or args.all:
        print("[CLI] Preprocessing data...")
        preprocess_data(DATA_DIR)

    if args.train or args.all:
        print(f"[CLI] Training model {args.model}...")
        train_model(args.model, DATA_DIR, config=args.config)

    if args.evaluate or args.all:
        print("[CLI] Evaluating model...")
        evaluate_model(DATA_DIR, args.model, args.config)

if __name__ == "__main__":
    main()
