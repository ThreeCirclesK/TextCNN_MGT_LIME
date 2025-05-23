import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_weights, lr=1e-4, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(train_weights, dtype=torch.float).to(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for input1, input2, labels, _ in loader:
            input1, input2, labels = input1.to(self.device), input2.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input1, input2)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        criterion = getattr(self, 'val_criterion', self.criterion)

        with torch.no_grad():
            for x1, x2, y, _ in loader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                out = self.model(x1, x2)
                loss = criterion(out, y)
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total if total else 0
        avg_loss = total_loss / len(loader)
        return avg_loss, acc

    def set_validation_weights(self, val_weights):
        self.val_criterion = nn.CrossEntropyLoss(weight=torch.tensor(val_weights, dtype=torch.float).to(self.device))

    def save_model(self, path="best_model.pt"):
        torch.save(self.model.state_dict(), path)

    def get_model(self):
        return self.model
