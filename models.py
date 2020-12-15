import torch.nn as nn

class LM(nn.Module):
    def __init__(self):
        super(LM, self).__init__()
        self.fc1 = nn.Linear(90, 1)

    def forward(self, x):
        output = self.fc1(x)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(90, 32)
        self.dropout= nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        h_1 = self.fc1(x)
        h_1 = self.dropout(h_1)
        h_1 = F.relu(h_1)
        output = self.fc2(h_1)
        return output
