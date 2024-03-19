import torch
import torch.nn as nn

class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NextWordPredictor, self).__init__()
        self.embedding = nn.Embedding(10000, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 10000)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        prediction = self.fc(output)
        return prediction
    
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        final_loss = running_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    return final_loss

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))
            running_loss += loss.item()
    return running_loss / len(test_loader)


