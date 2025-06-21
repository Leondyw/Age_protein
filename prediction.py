import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import Age_prediciton, normalize
from tqdm import tqdm

train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')

train_X, train_y, val_X, val_y, mean_y, std_y = normalize(train_df, val_df)
train_dataset = TensorDataset(train_X, train_y)
val_dataset = TensorDataset(val_X, val_y)
train_data = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=100, shuffle=False)

model = Age_prediciton(input_size=train_X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
num_epochs = 100

def train_model(model, train_data, val_data, criterion, optimizer, num_epochs):
    train_loss = []
    val_loss = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for X, y in train_data:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        with torch.no_grad():
            for X, y in val_data:
                val_loss.append(criterion(model(X), y).item())
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {(train_loss[-1])}, Val Loss: {(val_loss[-1])}")
    return model

if __name__ == "__main__":
    model = train_model(model, train_data, val_data, criterion, optimizer, num_epochs)