# dbn_with_dropout_earlystopping.py

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import joblib

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super().__init__()
        self.n_visible, self.n_hidden, self.k = n_visible, n_hidden, k
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
    def sample_from_p(self, p): return torch.bernoulli(p)
    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, self.sample_from_p(p_h)
    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, self.sample_from_p(p_v)
    def contrastive_divergence(self, v):
        v0 = v; h0_prob, _ = self.v_to_h(v0); vk = v0
        for _ in range(self.k):
            _, hk = self.v_to_h(vk)
            _, vk = self.h_to_v(hk)
        hk_prob, _ = self.v_to_h(vk)
        self.W.grad = torch.matmul(h0_prob.t(), v0) - torch.matmul(hk_prob.t(), vk)
        self.v_bias.grad = torch.sum(v0 - vk, dim=0)
        self.h_bias.grad = torch.sum(h0_prob - hk_prob, dim=0)
    def forward(self, v):
        h_prob, h_sample = self.v_to_h(v)
        for _ in range(self.k):
            v_prob, v_sample = self.h_to_v(h_sample)
            h_prob, h_sample = self.v_to_h(v_sample)
        return v, v_prob
    def train_rbm(self, loader, lr=0.001, epochs=10):
        for epoch in range(1, epochs+1):
            loss_sum = 0
            for v, in loader:
                v = v.view(-1, self.n_visible)
                self.zero_grad()
                self.contrastive_divergence(v)
                for p in self.parameters():
                    p.data += lr * p.grad
                loss_sum += torch.sum((v - self.forward(v)[1])**2).item()
            print(f"[RBM] Epoch {epoch}/{epochs} — Loss: {loss_sum:.4f}")

class DBN(nn.Module):
    def __init__(self, rbm1, rbm2, output_dim, dropout_p=0.3):
        super().__init__()
        self.fc1 = nn.Linear(rbm1.n_visible, rbm1.n_hidden)
        self.fc1.weight.data = rbm1.W.data.clone()
        self.fc1.bias.data = rbm1.h_bias.data.clone()
        self.drop1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(rbm2.n_visible, rbm2.n_hidden)
        self.fc2.weight.data = rbm2.W.data.clone()
        self.fc2.bias.data = rbm2.h_bias.data.clone()
        self.drop2 = nn.Dropout(dropout_p)
        self.out = nn.Linear(rbm2.n_hidden, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.drop1(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.drop2(x)
        return torch.sigmoid(self.out(x))

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs*targets + (1-inputs)*(1-targets)
        mod = (1-p_t)**self.gamma
        loss = self.alpha * mod * bce
        return loss.mean() if self.reduction=='mean' else loss.sum()

# 1. Load & augment train
df = pd.read_csv('train_data.csv')
X = df.drop(columns=['Label','Substance','Canonical_SMILES','formula']).values
y = df['Label'].values
X_aug, y_aug = SMOTE(random_state=42).fit_resample(X, y)

# 2. Scale & save scaler
scaler = StandardScaler().fit(X_aug)
joblib.dump(scaler, 'scaler.pkl')
X_scaled = scaler.transform(X_aug)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_aug, dtype=torch.float32).unsqueeze(1)

# 3. Pretrain RBMs
loader1 = DataLoader(TensorDataset(X_tensor), batch_size=32, shuffle=True)
rbm1 = RBM(n_visible=X_scaled.shape[1], n_hidden=64)
print(">>> Pretraining RBM-1")
rbm1.train_rbm(loader1)
with torch.no_grad():
    h1, _ = rbm1.v_to_h(X_tensor)
loader2 = DataLoader(TensorDataset(h1), batch_size=32, shuffle=True)
rbm2 = RBM(n_visible=64, n_hidden=128)
print(">>> Pretraining RBM-2")
rbm2.train_rbm(loader2)

# 4. Fine-tune DBN with early stopping
model = DBN(rbm1, rbm2, output_dim=1)
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_auc, trials = 0, 0
patience = 5

for epoch in range(1,51):
    model.train()
    train_loss = 0
    for xb, yb in DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True):
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # validation
    model.eval()
    dfv = pd.read_csv('val_data.csv')
    Xv = scaler.transform(dfv.drop(columns=['Label','Substance','Canonical_SMILES','formula']).values)
    yv = dfv['Label'].values
    Xv_t = torch.tensor(Xv, dtype=torch.float32)
    with torch.no_grad():
        pv = model(Xv_t).numpy().ravel()
    auc_v = auc(*roc_curve(yv, pv)[:2])
    print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val AUC {auc_v:.4f}")
    if auc_v>best_auc:
        best_auc, trials = auc_v, 0
        torch.save(model.state_dict(), 'best_dbn.pth')
    else:
        trials+=1
        if trials>=patience:
            print("Early stopping.")
            break

# 5. Evaluate on test
model.load_state_dict(torch.load('best_dbn.pth'))
model.eval()
dft = pd.read_csv('test_data.csv')
Xt = scaler.transform(dft.drop(columns=['Label','Substance','Canonical_SMILES','formula']).values)
yt = dft['Label'].values
Xt_t = torch.tensor(Xt, dtype=torch.float32)
with torch.no_grad():
    pt = model(Xt_t).numpy().ravel()
preds = (pt>0.5).astype(int)
print("Test AUC:", auc(*roc_curve(yt, pt)[:2]))
print("Test MCC:", matthews_corrcoef(yt, preds))
print("Confusion Matrix:\n", confusion_matrix(yt, preds))
