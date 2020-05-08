import torch.nn as nn
import torch.nn.functional as F
import pdb


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(768, 1000)
        self.fc2 = nn.Linear(1000, 768)
        # pdb.set_trace()
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        # pdb.set_trace()
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class LinearProject(nn.Module):
    def __init__(self):
        super(LinearProject, self).__init__()
        
        self.fc = nn.Linear(768, 768)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        return x
