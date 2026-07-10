import sys
sys.path.append('.')
from load_data import load_UT_HAR_data
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report

data = load_UT_HAR_data('../data')

X_train = data['X_train'].squeeze(1)  # (samples, 250, 90)
X_test = data['X_test'].squeeze(1)
y_train = (data['y_train'] == 1).long()
y_test = (data['y_test'] == 1).long()

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

class FallLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=90, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

model = FallLSTM()

class_counts = torch.bincount(y_train)
weights = 1.0 / class_counts.float()
weights = weights / weights.sum()
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

best_val_loss = float('inf')
best_state = None
batch_size = 64

for epoch in range(60):
    model.train()
    permutation = torch.randperm(X_tr.size(0))
    for i in range(0, X_tr.size(0), batch_size):
        idx = permutation[i:i+batch_size]
        batch_x, batch_y = X_tr[idx], y_tr[idx]

        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = criterion(val_out, y_val)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Val Loss: {val_loss.item():.4f}")

model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)

print("Accuracy:", accuracy_score(y_test, preds))
print("Recall on fall:", recall_score(y_test, preds))
print(classification_report(y_test, preds, target_names=['not_fall', 'fall']))