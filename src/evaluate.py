import torch
import argparse
import os
import json

def evaluate(model_path, data_path, metrics_path):
    # Load model and test data
    from train import SimpleCNN # Re-use the model class
    model = SimpleCNN()
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
    test_data = torch.load(os.path.join(data_path, 'test.pt'), weights_only=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    
    # Kubeflow Metrics format
    metrics = {
        'metrics': [{
            'name': 'accuracy',
            'numberValue': float(accuracy),
            'format': "PERCENTAGE",
        }]
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Accuracy: {accuracy}")
    if accuracy < 0.8: # Threshold
        raise Exception("Model accuracy too low!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--metrics_path', type=str, required=True)
    args = parser.parse_args()
    evaluate(args.model_path, args.data_path, args.metrics_path)