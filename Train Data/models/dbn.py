import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class RBMLayer(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBMLayer, self).__init__()
        # Initialize parameters
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        # Probability of hidden given visible
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h.bernoulli(), p_h

    def sample_v(self, h):
        # Probability of visible given hidden
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v.bernoulli(), p_v

    def contrastive_divergence(self, v, lr=0.0001, k=1):
        # v0: data (detach to stop Autograd)
        v0 = v.clone().detach()

        # Gibbs sampling k steps
        vk = v0
        for _ in range(k):
            h_k, _ = self.sample_h(vk)
            vk, _ = self.sample_v(h_k)
            vk = vk.detach()

        # Compute probabilities for gradient update
        p_h0 = torch.sigmoid(F.linear(v0, self.W, self.h_bias))
        p_hk = torch.sigmoid(F.linear(vk, self.W, self.h_bias))

        # Calculate parameter updates
        dW   = (p_h0.t() @ v0 - p_hk.t() @ vk) / v0.size(0)
        dvb  = (v0 - vk).mean(0)
        dhb  = (p_h0 - p_hk).mean(0)

        # Apply updates manually without Autograd
        with torch.no_grad():
            self.W.add_(lr * dW)  # L2 regularization will be added later in optimizer
            self.v_bias.add_(lr * dvb)
            self.h_bias.add_(lr * dhb)

        # Return reconstruction loss for monitoring
        return ((v0 - vk) ** 2).mean().item()

class DBN(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(DBN, self).__init__()
        # Stack multiple RBM layers (more layers than before)
        self.rbms = nn.ModuleList([
            RBMLayer(n_visible if i == 0 else n_hidden[i-1], nh)
            for i, nh in enumerate(n_hidden)
        ])
        # Final classifier with Dropout
        self.classifier = nn.Sequential(
            nn.Linear(n_hidden[-1], 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Dropout layer with 50% probability
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass input through RBM encoders
        for rbm in self.rbms:
            _, x = rbm.sample_h(x)
        return self.classifier(x)

    def train_model(self, X_train, y_train, val_data=None, epochs=100, batch_size=64, lr=0.0001, weight_decay=1e-5):
        logging.info("Starting DBN pre-training...")
        # Pre-training each RBM layer with manual updates
        x = X_train
        for idx, rbm in enumerate(self.rbms):
            logging.info(f"Pre-training RBM layer {idx+1}/{len(self.rbms)}")
            for epoch in range(epochs):
                perm = torch.randperm(x.size(0))
                total_loss = 0.0
                for i in range(0, x.size(0), batch_size):
                    batch = x[perm[i:i+batch_size]]
                    loss = rbm.contrastive_divergence(batch, lr=lr)
                    total_loss += loss
                if epoch % 10 == 0:
                    avg_loss = total_loss / (x.size(0)/batch_size)
                    logging.info(f"RBM Layer {idx+1}, Epoch {epoch}, Loss: {avg_loss:.4f}")
            # Transform data for next layer
            _, x = rbm.sample_h(x)

        logging.info("Starting DBN fine-tuning...")
        # Fine-tuning with standard optimizer and backprop (with L2 regularization)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)  # L2 Regularization
        loss_fn = nn.BCELoss()

        for epoch in range(epochs):
            perm = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                xb = X_train[perm[i:i+batch_size]]
                yb = y_train[perm[i:i+batch_size]].unsqueeze(1)
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
                    logging.info(f"Fine-tune Epoch {epoch}, Validation Accuracy: {acc:.4f}")
