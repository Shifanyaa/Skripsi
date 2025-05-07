import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os

# ---------- RBM Class ----------
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, self.sample_from_p(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, self.sample_from_p(p_v)

    def forward(self, v):
        h_prob, h_sample = self.v_to_h(v)
        for _ in range(self.k):
            v_prob, v_sample = self.h_to_v(h_sample)
            h_prob, h_sample = self.v_to_h(v_sample)
        return v, v_prob

    def contrastive_divergence(self, v):
        v0 = v
        h0_prob, h0_sample = self.v_to_h(v0)
        vk = v0
        for _ in range(self.k):
            hk_prob, hk_sample = self.v_to_h(vk)
            vk_prob, vk = self.h_to_v(hk_sample)
        hk_prob, _ = self.v_to_h(vk)
        self.W.grad = torch.matmul(h0_prob.t(), v0) - torch.matmul(hk_prob.t(), vk)
        self.v_bias.grad = torch.sum(v0 - vk, dim=0)
        self.h_bias.grad = torch.sum(h0_prob - hk_prob, dim=0)

    def train_rbm(self, train_loader, lr=0.01, epochs=10):
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                v, = batch
                v = v.view(-1, self.n_visible)
                self.zero_grad()
                self.contrastive_divergence(v)
                for param in self.parameters():
                    param.data += lr * param.grad
                epoch_loss += torch.sum((v - self.forward(v)[1])**2)
            print(f"[RBM] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.item():.4f}")


# ---------- DBN Class ----------
class DBN(nn.Module):
    def __init__(self, rbm1, rbm2, output_dim):
        super(DBN, self).__init__()
        self.fc1 = nn.Linear(rbm1.n_visible, rbm1.n_hidden)
        self.fc1.weight.data = rbm1.W.data
        self.fc1.bias.data = rbm1.h_bias.data

        self.fc2 = nn.Linear(rbm2.n_visible, rbm2.n_hidden)
        self.fc2.weight.data = rbm2.W.data
        self.fc2.bias.data = rbm2.h_bias.data

        self.out = nn.Linear(rbm2.n_hidden, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x


# ---------- Data Handling ----------
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Substance', 'Canonical_SMILES', 'formula'], errors='ignore')
    X = df.drop('Label', axis=1).values
    y = df['Label'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def prepare_tensors(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X_tensor, y_tensor


# ---------- Fine-tuning ----------
def fine_tune(dbn_model, train_loader, epochs=20, lr=0.1):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(dbn_model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            output = dbn_model(x_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[DBN] Epoch {epoch+1}/{epochs}, Fine-tune Loss: {total_loss:.4f}")


# ---------- Evaluation ----------
def evaluate(model, X_test, y_test):
    y_pred_prob = model(X_test).detach().numpy().ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    print("\n[Evaluation Results]")
    print("Matthews Correlation Coefficient:", mcc)
    print("Confusion Matrix:\n", cm)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.savefig("roc_curve.png")
    print("ROC curve saved to 'roc_curve.png'")


# ---------- Main Pipeline ----------
def main():
    # Load & preprocess
    X, y = load_and_preprocess('train_data.csv')
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train_tensor, y_train_tensor = prepare_tensors(X_train, y_train)
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)

    # Train RBM layers
    rbm1 = RBM(n_visible=X_train.shape[1], n_hidden=256)
    rbm1.train_rbm(train_loader, lr=0.01, epochs=10)
    with torch.no_grad():
        h1, _ = rbm1.v_to_h(X_train_tensor)

    rbm2 = RBM(n_visible=256, n_hidden=128)
    rbm2.train_rbm(DataLoader(TensorDataset(h1), batch_size=32, shuffle=True), lr=0.01, epochs=10)

    # Fine-tune DBN
    dbn = DBN(rbm1, rbm2, output_dim=1)
    fine_tune(dbn, DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True), epochs=20)

    # Evaluate on test
    X_test_tensor, _ = prepare_tensors(X_test, y_test)
    evaluate(dbn, X_test_tensor, y_test)


if __name__ == '__main__':
    main()
