"""PyTorch LSTM/GRU sequence model for ordinal fatigue prediction."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from modeling.config import RANDOM_STATE
from modeling.data import SequenceData


class _RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, rnn_type, num_classes=6):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if isinstance(self.rnn, nn.LSTM):
            _, (hidden, _) = self.rnn(packed)
        else:
            _, hidden = self.rnn(packed)
        last = hidden[-1]
        return self.head(last)


class SequenceClassifier(BaseEstimator):
    def __init__(
        self,
        rnn_type="lstm",
        hidden_size=64,
        num_layers=1,
        dropout=0.2,
        lr=1e-3,
        batch_size=64,
        epochs=40,
        patience=8,
        val_fraction=0.15,
        max_grad_norm=1.0,
        random_state=RANDOM_STATE,
    ):
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.max_grad_norm = max_grad_norm
        self.random_state = random_state
        self.model_ = None
        self.device_ = torch.device("cpu")
        self.feature_mean_ = None
        self.feature_std_ = None

    def _prepare(self, X):
        if isinstance(X, SequenceData):
            return X.X, X.lengths, X.y, X.groups
        raise TypeError("SequenceClassifier expects SequenceData input.")

    def _scale_features(self, X_arr, lengths, fit=False):
        if fit:
            sums = np.zeros(X_arr.shape[2], dtype=np.float64)
            sq_sums = np.zeros(X_arr.shape[2], dtype=np.float64)
            count = 0
            for i, length in enumerate(lengths):
                chunk = X_arr[i, :length]
                sums += chunk.sum(axis=0)
                sq_sums += np.square(chunk).sum(axis=0)
                count += length
            mean = sums / max(count, 1)
            var = sq_sums / max(count, 1) - np.square(mean)
            std = np.sqrt(np.maximum(var, 1e-8))
            self.feature_mean_ = mean.astype(np.float32)
            self.feature_std_ = std.astype(np.float32)

        scaled = X_arr.copy()
        for i, length in enumerate(lengths):
            scaled[i, :length] = (scaled[i, :length] - self.feature_mean_) / self.feature_std_
        return scaled

    def _class_weights(self, y_arr):
        classes = np.arange(6)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_arr)
        return torch.tensor(weights, dtype=torch.float32, device=self.device_)

    def _train_epoch(self, loader, optimizer, criterion):
        self.model_.train()
        total_loss = 0.0
        n_batches = 0
        for batch_x, batch_len, batch_y in loader:
            batch_x = batch_x.to(self.device_)
            batch_len = batch_len.to(self.device_)
            batch_y = batch_y.to(self.device_)
            optimizer.zero_grad()
            logits = self.model_(batch_x, batch_len)
            loss = criterion(logits, batch_y)
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.model_.parameters(), self.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _eval_loss(self, loader, criterion):
        self.model_.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch_x, batch_len, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_len = batch_len.to(self.device_)
                batch_y = batch_y.to(self.device_)
                logits = self.model_(batch_x, batch_len)
                loss = criterion(logits, batch_y)
                total_loss += float(loss.item())
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _make_loader(self, X_arr, lengths, y_arr, shuffle):
        x_tensor = torch.tensor(X_arr, dtype=torch.float32)
        len_tensor = torch.tensor(lengths, dtype=torch.int64)
        y_tensor = torch.tensor(y_arr, dtype=torch.long)
        dataset = TensorDataset(x_tensor, len_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X, y=None):
        del y
        X_arr, lengths, y_arr, groups = self._prepare(X)
        X_arr = self._scale_features(X_arr, lengths, fit=True)

        torch.manual_seed(self.random_state)
        n_features = X_arr.shape[2]
        self.model_ = _RNNModule(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            rnn_type=self.rnn_type,
        ).to(self.device_)

        class_weights = self._class_weights(y_arr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        indices = np.arange(len(y_arr))
        if len(np.unique(groups)) >= 2 and self.val_fraction > 0:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.val_fraction,
                random_state=self.random_state,
            )
            train_idx, val_idx = next(splitter.split(indices, groups=groups))
        else:
            train_idx, val_idx = indices, np.array([], dtype=int)

        train_loader = self._make_loader(
            X_arr[train_idx], lengths[train_idx], y_arr[train_idx], shuffle=True
        )
        val_loader = None
        if len(val_idx) > 0:
            val_loader = self._make_loader(
                X_arr[val_idx], lengths[val_idx], y_arr[val_idx], shuffle=False
            )

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        best_state = None
        best_val_loss = float("inf")
        patience_left = self.patience

        for _ in range(self.epochs):
            self._train_epoch(train_loader, optimizer, criterion)
            if val_loader is None:
                continue
            val_loss = self._eval_loss(val_loader, criterion)
            if val_loss + 1e-4 < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        X_arr, lengths, _, _ = self._prepare(X)
        X_arr = self._scale_features(X_arr, lengths, fit=False)
        self.model_.eval()
        preds = []
        x_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device_)
        len_tensor = torch.tensor(lengths, dtype=torch.int64).to(self.device_)
        with torch.no_grad():
            for start in range(0, len(X_arr), self.batch_size):
                end = start + self.batch_size
                logits = self.model_(x_tensor[start:end], len_tensor[start:end])
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds)


def build_lstm(
    rnn_type="lstm",
    hidden_size=64,
    num_layers=1,
    dropout=0.2,
    lr=1e-3,
    batch_size=64,
    epochs=40,
    patience=8,
    **kwargs,
):
    del kwargs
    return SequenceClassifier(
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        random_state=RANDOM_STATE,
    )
