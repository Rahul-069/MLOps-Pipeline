# import torch
# import argparse
# import os
# import json
# from train import CNN # Re-use the model class

# def evaluate(model_path, data_path, metrics_path):
#     # Load model and test data
    
#     model = CNN()
#     model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
#     test_data = torch.load(os.path.join(data_path, 'test.pt'), weights_only=False)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
    
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
    
#     accuracy = correct / len(test_loader.dataset)
    
#     # Kubeflow Metrics format
#     metrics = {
#         'metrics': [{
#             'name': 'accuracy',
#             'numberValue': float(accuracy),
#             'format': "PERCENTAGE",
#         }]
#     }
#     with open(metrics_path, 'w') as f:
#         json.dump(metrics, f)
    
#     print(f"Accuracy: {accuracy}")
#     if accuracy < 0.8: # Threshold
#         raise Exception("Model accuracy too low!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--data_path', type=str, required=True)
#     parser.add_argument('--metrics_path', type=str, required=True)
#     args = parser.parse_args()
#     evaluate(args.model_path, args.data_path, args.metrics_path)










import torch
import argparse
import os
import json
from torch.utils.data import TensorDataset, DataLoader
from train import CNN # Ensure train.py is in the same directory or PYTHONPATH

def evaluate(model_path, data_path, metrics_path):
    # 1. Load the Model
    model = CNN()
    # Use map_location='cpu' to ensure it works even if trained on GPU
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location='cpu'))
    
    # 2. Unpack the Test Data (Tuple of Tensors)
    # This must match the (images, labels) format from your preprocess.py
    images, labels = torch.load(os.path.join(data_path, 'test.pt'), weights_only=False)
    
    # 3. Create DataLoader
    test_dataset = TensorDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 4. Evaluation Loop
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    
    # 5. Save Kubeflow Metrics
    # In KFP v2, writing this JSON is the "standard" way for UI display
    metrics = {
        'metrics': [{
            'name': 'accuracy',
            'numberValue': float(accuracy),
            'format': "PERCENTAGE",
        }]
    }
    
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Evaluation complete. Accuracy: {accuracy:.2%}")
    
    # 6. Safety Threshold
    if accuracy < 0.8:
        raise Exception(f"Model accuracy ({accuracy:.2%}) is below the 80% threshold!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--metrics_path', type=str, required=True)
    args = parser.parse_args()
    evaluate(args.model_path, args.data_path, args.metrics_path)