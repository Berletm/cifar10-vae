import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optimizers

# VGG16 architecture
class CNNClassificator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.2))
                                    
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.2))
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.2))

        self.flatten = nn.Flatten()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(in_features=512, out_features=512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(in_features=512, out_features=10))
        
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) # 32 x 32 x 64
        x = self.pool(x)  # 16 x 16 x 64

        x = self.conv2(x) # 16 x 16 x 128
        x = self.pool(x)  # 8 x 8 x 128

        x = self.conv3(x) # 8 x 8 x 256
        x = self.pool(x)  # 4 x 4 x 256

        x = self.conv4(x) # 4 x 4 x 512
        x = self.pool(x)  # 2 x 2 x 512

        x = self.conv5(x)  # 2 x 2 x 512
        x = self.pool(x)  # 1 x 1 x 512

        x = self.flatten(x) # 1*1*512

        x = self.fc(x) # 1 x 10

        return x

    def predict(self, x: torch.Tensor) ->  torch.Tensor:
        return self.softmax(self.forward(x))
    

def train(n_epoch: int, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
    optimizer = optimizers.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(n_epoch):    
        total   = 0
        correct = 0
        train_loss = 0.0
        model.train()
        for imgs, labels in train_loader:
            labels = labels.to(device)
            imgs   = imgs.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            train_loss += loss.item()
            total   += len(labels)
            preds = torch.softmax(logits, dim=0)
            correct += (preds.argmax(dim=1) == labels).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = correct / total
        train_loss = train_loss / total

        total   = 0
        correct = 0
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                labels = labels.to(device)
                imgs   = imgs.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                total   += len(labels)
                preds = torch.softmax(logits, dim=0)
                correct += (preds.argmax(dim=1) == labels).long().sum()

            val_acc = correct / total
            val_loss = val_loss / total

        print(f"Epoch {epoch + 1} / {n_epoch} | Val loss {val_loss:.4f} | Val acc {val_acc:.4f} | Train loss {train_loss:.4f} | Train acc {train_acc:.4f}")

    return model