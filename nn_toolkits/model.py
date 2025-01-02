import torch as t
import torch.nn as nn
import torch.nn.functional as f


class GRUBlock(nn.Module):
    def __init__(self, in_size, h_size, n_layers):
        super(GRUBlock, self).__init__()

        self.gru = nn.GRU(in_size, h_size, n_layers, batch_first=True, dropout=0.1)

        self.fc1 = nn.Linear(h_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)

        self.ln1 = nn.LayerNorm(h_size)
        self.ln2 = nn.LayerNorm(h_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # GRU Layer
        out_gru, *_ = self.gru(x)

        # Add & Norm
        out = self.ln1(out_gru + x)

        # FFN Layers
        out_fc = f.relu(self.fc1(out))
        out_fc = f.relu(self.fc2(out_fc))

        # Add & Norm + Dropout
        out = self.ln2(out_fc + out)
        out = self.dropout(out)

        return out


class TSModel(nn.Module):
    def __init__(self, in_size, h_size, out_size, n_layers, n_blocks):
        super(TSModel, self).__init__()

        self.fc_in = nn.Linear(in_size, h_size)

        self.blocks = nn.ModuleList([GRUBlock(h_size, h_size, n_layers) for _ in range(n_blocks)])

        self.fc1 = nn.Linear(h_size, h_size)
        self.fc2 = nn.Linear(h_size, out_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Один полносвязный слой, чтобы подогнать данные под нужный размер
        out = f.relu(self.fc_in(x))

        # GRU Blocks
        for block in self.blocks:
            out = block(out)

        # Dropout
        out = self.dropout(out)

        # FFN Layers
        out = f.relu(self.fc1(out))
        out = self.fc2(out)

        return out


class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    @staticmethod
    def forward(y_pred, y_true):
        log_pred = t.log1p(t.clamp(y_pred, min=0))
        log_true = t.log1p(t.clamp(y_true, min=0))

        loss = t.mean((log_pred - log_true) ** 2)

        rmsle = t.sqrt(loss)
        return rmsle
