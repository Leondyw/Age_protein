import torch
from torch import nn

def normalize(train_df, val_df=None, test_df=None):
    train_X = train_df.iloc[:,:-1]
    train_y = train_df.iloc[:,-1]
    mean_y = train_y.mean()
    std_y = train_y.std()
    train_y = (train_y - mean_y) / std_y
    train_X = torch.tensor(train_X.values, dtype=torch.float32)
    train_y = torch.tensor(train_y.values, dtype=torch.float32)

    if val_df is not None:
        val_X = val_df.iloc[:,:-1]
        val_y = val_df.iloc[:,-1]
        val_y = (val_y - mean_y) / std_y
        val_X = torch.tensor(val_X.values, dtype=torch.float32)
        val_y = torch.tensor(val_y.values, dtype=torch.float32)
        
    elif test_df is not None:
        test_X = test_df.iloc[:,:-1]
        test_y = test_df.iloc[:,-1]
        test_y = (test_y - mean_y) / std_y
        test_X = torch.tensor(test_X.values, dtype=torch.float32)
        test_y = torch.tensor(test_y.values, dtype=torch.float32)
        
    if val_df is not None and test_df is not None:
        return train_X, train_y, val_X, val_y, test_X, test_y, mean_y, std_y
    elif val_df is not None:
        return train_X, train_y, val_X, val_y, mean_y, std_y
    elif test_df is not None:
        return train_X, train_y, test_X, test_y, mean_y, std_y
    else:
        return train_X, train_y, mean_y, std_y
    
class Age_prediciton(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        self.init_weights()

    def forward(self, x):
        x = self.MLP(x)
        return x.squeeze(1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    
if __name__ == "__main__":
    model = Age_prediciton(input_size=100)
    data = torch.randn(3, 100)
    print(model(data))