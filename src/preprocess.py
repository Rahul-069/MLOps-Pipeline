import torch
from torchvision import datasets, transforms
import argparse
import os

def preprocess(output_path):
    os.makedirs(output_path, exist_ok=True)
    # Simple normalization for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    torch.save(train_set, os.path.join(output_path, 'train.pt'))
    torch.save(test_set, os.path.join(output_path, 'test.pt'))
    print(f"Data preprocessed and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    preprocess(args.output_path)
