# import torch
# from torchvision import datasets, transforms
# import argparse
# import os

# def preprocess(output_path):
#     os.makedirs(output_path, exist_ok=True)
#     # Simple normalization for MNIST
#     transform = transforms.Compose([
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#     train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
#     test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
#     torch.save(train_set, os.path.join(output_path, 'train.pt'))
#     torch.save(test_set, os.path.join(output_path, 'test.pt'))
#     print(f"Data preprocessed and saved to {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--output_path', type=str, required=True)
#     args = parser.parse_args()
#     preprocess(args.output_path)

import torch
import boto3
import argparse
import os
from torchvision import transforms

def download_from_minio(bucket, object_name, local_path):
    s3 = boto3.client('s3',
                      endpoint_url='http://172.21.64.1:9000',
                      aws_access_key_id='minio',
                      aws_secret_access_key='minio123')
    s3.download_file(bucket, object_name, local_path)

def preprocess(bucket_name, data_output_path):
    os.makedirs(data_output_path, exist_ok=True)
    
    # 1. Download the packed .pt files
    tmp_train = "/tmp/train.pt"
    tmp_test = "/tmp/test.pt"
    download_from_minio(bucket_name, 'train.pt', tmp_train)
    download_from_minio(bucket_name, 'test.pt', tmp_test)

    # 2. Unpack the Tensors
    # Based on your packing code, these are tuples: (images, labels)
    train_images, train_labels = torch.load(tmp_train)
    test_images, test_labels = torch.load(tmp_test)

    # 3. Apply ONLY Normalization
    # Since images are already [0, 1] tensors, we apply the stats directly
    # MNIST Stats: Mean=0.1307, Std=0.3081
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    
    # Apply normalization to the image tensors
    # train_images shape is (N, 1, 28, 28)
    norm_train_images = normalize(train_images)
    norm_test_images = normalize(test_images)

    # 4. Save the fully preprocessed objects for the Trainer
    torch.save((norm_train_images, train_labels), os.path.join(data_output_path, 'train.pt'))
    torch.save((norm_test_images, test_labels), os.path.join(data_output_path, 'test.pt'))
    
    print("Normalization applied. Data saved to artifacts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    preprocess(args.bucket, args.output_path)
