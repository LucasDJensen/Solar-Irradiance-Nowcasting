import torch
from torch import nn


# Define the simple LSTM model from earlier
class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 1, num_layers: int = 2, dropout: float = 0.3):
        super(SimpleLSTM, self).__init__()
        # 1D‐Conv frontend to capture local patterns
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm1d(32)
        self.drop_conv = nn.Dropout(dropout)
        # first LSTM block
        self.lstm1 = nn.LSTM(32, hidden_size, batch_first=True, dropout=dropout, bidirectional=False)
        self.ln1   = nn.LayerNorm(hidden_size)
        self.drop1 = nn.Dropout(dropout)

        # second LSTM block
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout, bidirectional=False)
        self.ln2   = nn.LayerNorm(hidden_size)
        self.drop2 = nn.Dropout(dropout)
        # self‐attention block
        # if you bump hidden_size to 64, you can go to 8 heads here
        n_heads = 4 if hidden_size < 64 else 8
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, dropout=dropout)

        self.ln_attn = nn.LayerNorm(hidden_size)
        self.drop_attn = nn.Dropout(dropout)
        # time‐distributed linear head
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # conv frontend: x=(B, T, F) -> (B, F, T) -> conv -> (B,16,T) -> (B, T,16)
        xc = x.permute(0,2,1)
        xc = self.conv1(xc)
        xc = self.bn_conv1(xc)
        xc = torch.relu(xc)
        xc = self.drop_conv(xc)
        xc = xc.permute(0, 2, 1)

        # 1st LSTM block
        out, _ = self.lstm1(xc)
        out = self.ln1(out)
        out = torch.relu(out)
        out = self.drop1(out)
        skip = out

        # 2nd LSTM block + residual
        out, _ = self.lstm2(out)
        out = self.ln2(out + skip)  # add skip connection
        out = torch.relu(out)
        out = self.drop2(out)

        # self‐attention across time steps
        # MultiheadAttention expects (T, B, C)
        o_t = out.permute(1,0,2)
        attn_out, _ = self.attn(o_t, o_t, o_t)
        out = self.ln_attn(attn_out + o_t)
        out = self.drop_attn(out).permute(1, 0, 2)

        # per‐time‐step prediction
        return self.fc(out)
