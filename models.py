import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin

from utils import set_seed


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class AutoEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, epochs=20, batch_size=256, lr=1e-3, verbose=1, device=None, seed=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.seed = seed

    def fit(self, X, y=None):
        set_seed(self.seed)
        X = np.array(X, dtype=np.float32)
        self.input_dim = X.shape[1]
        self.model = Autoencoder(self.input_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(loader.dataset)
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.6f}")
        return self

    # function that returns the error of reconstruction (anomaly score)
    def score_samples(self, X):
        X = np.array(X, dtype=np.float32)
        with torch.no_grad():
            inputs = torch.from_numpy(X).to(self.device)
            outputs = self.model(inputs).cpu().numpy()
        errors = np.mean((X - outputs) ** 2, axis=1)
        return errors

    def transform(self, X):
        return self.score_samples(X)

    
    # return binary labels according to a threshold (default : percentile 99)
    def predict(self, X, threshold=None):
        scores = self.score_samples(X)
        if threshold is None:
            threshold = np.percentile(scores, 99)
        return (scores > threshold).astype(int)