import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * input_dim, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.out(x))

    def train_model(self, X_train, y_train, val_data=None, epochs=50, batch_size=64, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        logging.info("Starting CNN training...")
        for epoch in range(epochs):
            perm = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                xb = X_train[perm[i:i + batch_size]]
                yb = y_train[perm[i:i + batch_size]].unsqueeze(1)
                out = self.forward(xb)
                loss = loss_fn(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0 and val_data is not None:
                val_x, val_y = val_data
                with torch.no_grad():
                    preds = self.forward(val_x).squeeze()
                    pred_bin = (preds > 0.5).float()
                    acc = (pred_bin == val_y).float().mean().item()
                    logging.info(f"Epoch {epoch}, Validation Accuracy: {acc:.4f}")
