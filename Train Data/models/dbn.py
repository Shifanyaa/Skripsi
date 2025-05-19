# models/dbn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class RBMLayer(nn.Module):
    def __init__(self, n_visible, n_hidden, momentum=0.5):
        super(RBMLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.W_mom = torch.zeros_like(self.W)
        self.vb_mom = torch.zeros_like(self.v_bias)
        self.hb_mom = torch.zeros_like(self.h_bias)
        self.momentum = momentum

    def sample_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h.bernoulli(), p_h

    def sample_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v.bernoulli(), p_v

    def contrastive_divergence(self, v, lr=1e-4, k=1):
        v0 = v.detach()
        vk = v0
        for _ in range(k):
            h_k, _ = self.sample_h(vk)
            vk, _ = self.sample_v(h_k)
            vk = vk.detach()

        p_h0 = torch.sigmoid(F.linear(v0, self.W, self.h_bias))
        p_hk = torch.sigmoid(F.linear(vk, self.W, self.h_bias))

        dW = (p_h0.t() @ v0 - p_hk.t() @ vk) / v0.size(0)
        dvb = (v0 - vk).mean(0)
        dhb = (p_h0 - p_hk).mean(0)

        # momentum update
        self.W_mom = self.momentum * self.W_mom + lr * dW
        self.vb_mom = self.momentum * self.vb_mom + lr * dvb
        self.hb_mom = self.momentum * self.hb_mom + lr * dhb

        with torch.no_grad():
            self.W.add_(self.W_mom)
            self.v_bias.add_(self.vb_mom)
            self.h_bias.add_(self.hb_mom)

        return ((v0 - vk) ** 2).mean().item()


class DBN(nn.Module):
    def __init__(self, n_visible, hidden_sizes=[512,256,128]):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList([
            RBMLayer(n_visible if i==0 else hidden_sizes[i-1], h)
            for i,h in enumerate(hidden_sizes)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for rbm in self.rbms:
            _, x = rbm.sample_h(x)
        return self.classifier(x)

    def train_model(self, X_train, y_train, val_data=None,
                    epochs=100, batch_size=128, lr=1e-4,
                    weight_decay=1e-4, clip_grad=1.0, patience=10):
        # --- Pre-training RBMs (sama seperti sebelumnya) ---
        logging.info("Starting DBN pre-training...")
        x = X_train
        for idx, rbm in enumerate(self.rbms):
            logging.info(f"Pre-training RBM layer {idx+1}/{len(self.rbms)}")
            for ep in range(epochs):
                perm = torch.randperm(x.size(0))
                running_loss = 0.0
                for i in range(0, x.size(0), batch_size):
                    batch = x[perm[i:i+batch_size]]
                    loss = rbm.contrastive_divergence(batch, lr=lr)
                    running_loss += loss
                if ep % 10 == 0:
                    avg = running_loss / (x.size(0)/batch_size)
                    logging.info(f" RBM{idx+1} Epoch {ep}: Loss={avg:.4f}")
            _, x = rbm.sample_h(x)

        # --- Fine-tuning Supervised ---
        logging.info("Starting DBN fine-tuning...")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # Perhalus scheduler dengan patience=3
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3)
        criterion = nn.BCELoss()

        # List untuk plotting
        self.train_losses, self.train_accs = [], []
        self.val_losses,   self.val_accs   = [], []

        best_val = float('inf')
        wait = 0

        for ep in range(epochs):
            # Training loop...
            self.train()
            perm = torch.randperm(X_train.size(0))
            total_loss, correct, total = 0, 0, 0
            for i in range(0, X_train.size(0), batch_size):
                xb = X_train[perm[i:i+batch_size]]
                yb = y_train[perm[i:i+batch_size]].unsqueeze(1)
                out = self.forward(xb)
                loss = criterion(out, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                pred = (out>0.5).float()
                correct += (pred==yb).sum().item()
                total += xb.size(0)
            train_loss = total_loss/total
            train_acc  = correct/total
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation
            if val_data is not None:
                self.eval()
                val_x, val_y = val_data
                with torch.no_grad():
                    out_v = self.forward(val_x)
                    loss_v = criterion(out_v, val_y.unsqueeze(1)).item()
                    pred_v = (out_v>0.5).float()
                    acc_v  = (pred_v==val_y.unsqueeze(1)).float().mean().item()
                self.val_losses.append(loss_v)
                self.val_accs.append(acc_v)

                scheduler.step(loss_v)
                logging.info(f"Epoch {ep}: Train(l={train_loss:.4f},a={train_acc:.4f}) | Val(l={loss_v:.4f},a={acc_v:.4f})")

                # Early stopping
                if loss_v < best_val:
                    best_val = loss_v
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        logging.info("Early stopping.")
                        break
