import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SimpleNN(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, h3_size):
        super(SimpleNN, self).__init__()
        # define layers
        self.input_layer = nn.Linear(input_size, h1_size)
        self.h1 = nn.Linear(h1_size, h2_size)
        self.h2 = nn.Linear(h2_size, h3_size)
        self.output_layer = nn.Linear(h3_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # define forward pass
        x = self.relu(self.input_layer(x))
        x = self.relu(self.h1(x))
        x = self.relu(self.h2(x))
        x = self.output_layer(x)
        return x

def main():
    '''
    making a basic neural network
    '''
    # data basic
    df = pd.read_csv('processed_airbnb.csv')
    X = df.drop('review_scores_rating', axis=1).values
    y = df['review_scores_rating'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    # hyperparameters
    input_size = X_train.shape[1]
    h1_size = 64
    h2_size = 32
    h3_size = 16
    learning_rate = 0.001
    n_epochs = 20

    # model initialization
    model = SimpleNN(input_size, h1_size, h2_size, h3_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test), y_test)
            print(f'Epoch {epoch+1}/{n_epochs} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}')

    return

if __name__ == "__main__":
    main()
