# import torch
# import torch.nn as nn
# import torch.optim as optim
# import argparse
# import os

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, 128),
#             nn.ReLU(),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# def train(input_path, model_path, epochs=5):
#     os.makedirs(model_path, exist_ok=True)

#     train_data = torch.load(os.path.join(input_path, 'train.pt'), weights_only=False)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

#     model = CNN()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#     criterion = nn.CrossEntropyLoss()   # ✅ FIXED

#     model.train()
#     for epoch in range(epochs):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             optimizer.zero_grad()

#             output = model(data)
#             loss = criterion(output, target)  # ✅ FIXED

#             loss.backward()
#             optimizer.step()

#     torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))
#     print("Training complete.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_path', type=str, required=True)
#     parser.add_argument('--model_path', type=str, required=True)
#     args = parser.parse_args()
#     train(args.input_path, args.model_path)









import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train(input_path, model_path, epochs=5):
    os.makedirs(model_path, exist_ok=True)

    # 1. Load the tuple and wrap it
    images, labels = torch.load(os.path.join(input_path, 'train.pt'), weights_only=False)
    train_dataset = torch.utils.data.TensorDataset(images, labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN()
    # Note: lr=0.01 is quite high for Adam; 0.001 is usually safer!
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))
    print("Training complete and model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    train(args.input_path, args.model_path)