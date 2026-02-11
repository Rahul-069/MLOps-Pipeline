import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1 = nn.Linear(1440, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = x.view(-1, 1440)
        return torch.log_softmax(self.fc1(x), dim=1)

def train(input_path, model_path, epochs=1):
    os.makedirs(model_path, exist_ok=True)
    train_data = torch.load(os.path.join(input_path, 'train.pt'), weights_only=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    model = SimpleCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    train(args.input_path, args.model_path)