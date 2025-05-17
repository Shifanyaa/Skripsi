import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import os

# --- RBM ---
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

    def train_rbm(self, train_loader, lr=0.01, epochs=100):
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
            print(f"RBM Epoch {epoch+1}, Loss: {epoch_loss.item():.4f}")

# --- DBN ---
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

# --- Load CSVs and Preprocess ---
def load_and_preprocess(file, scaler=None, fit=False):
    df = pd.read_csv(file)
    df = df.drop(columns=['Substance', 'Canonical_SMILES', 'formula'], errors='ignore')
    X = df.drop('Label', axis=1).values
    y = df['Label'].values
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1), X_scaled, y

scaler = MinMaxScaler()
X_train, y_train, X_train_np, y_train_np = load_and_preprocess("train_data_balanced.csv", scaler, fit=True)
X_val, y_val, X_val_np, y_val_np = load_and_preprocess("val_data.csv", scaler)
X_test, y_test, X_test_np, y_test_np = load_and_preprocess("test_data.csv", scaler)

# --- Handle Class Imbalance using RandomOverSampler ---
ros = RandomOverSampler(random_state=42)
X_train_np_ros, y_train_np_ros = ros.fit_resample(X_train_np, y_train_np)

# Convert to Torch tensors
X_train_ros = torch.tensor(X_train_np_ros, dtype=torch.float32)
y_train_ros = torch.tensor(y_train_np_ros, dtype=torch.float32).unsqueeze(1)

# --- Pretrain RBMs ---
train_loader = DataLoader(TensorDataset(X_train_ros), batch_size=32, shuffle=True)
rbm1 = RBM(n_visible=X_train_ros.shape[1], n_hidden=256)
rbm1.train_rbm(train_loader, lr=0.01, epochs=100)

with torch.no_grad():
    h1, _ = rbm1.v_to_h(X_train_ros)

rbm2 = RBM(n_visible=256, n_hidden=128)
rbm2.train_rbm(DataLoader(TensorDataset(h1), batch_size=32, shuffle=True), lr=0.01, epochs=10)

# --- Fine-tune DBN ---
model = DBN(rbm1, rbm2, output_dim=1)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_sup_loader = DataLoader(TensorDataset(X_train_ros, y_train_ros), batch_size=32, shuffle=True)

for epoch in range(20):
    total_loss = 0
    for x_batch, y_batch in train_sup_loader:
        output = model(x_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Fine-tuning Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --- Evaluation Function ---
def evaluate(X, y, name):
    y_prob = model(X).detach().numpy().ravel()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = y.numpy().ravel()
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n{name} Evaluation:")
    print("MCC:", mcc)
    print("Confusion Matrix:\n", cm)

    plt.figure()
    plt.plot(fpr, tpr, label=f'{name} ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend()
    plt.grid()
    plt.savefig(f"{name.lower()}_roc_curve.png")
    plt.close()

    # Export evaluation results
    pd.DataFrame({
        "True_Label": y_true,
        "Pred_Prob": y_prob,
        "Pred_Label": y_pred
    }).to_csv(f"{name.lower()}_predictions.csv", index=False)

# --- Run Evaluation ---
evaluate(X_val, y_val, "Validation")
evaluate(X_test, y_test, "Test")

# --- Baseline Comparison ---
print("\nBaseline Model Comparison")
for name, clf in zip(["Logistic Regression", "Random Forest"],
                     [LogisticRegression(max_iter=1000), RandomForestClassifier()]):
    clf.fit(X_train_np_ros, y_train_np_ros)
    score = clf.score(X_test_np, y_test_np)
    print(f"{name} Accuracy: {score:.4f}")

# --- Cross-validation ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nCross-validation with Logistic Regression")
cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X_train_np_ros, y_train_np_ros, cv=skf, scoring='accuracy')
print("Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
