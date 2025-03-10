import torch
from torch import nn

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden = hidden.permute(1, 0, 2).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size)
        attn_input = torch.cat((hidden, encoder_outputs), dim=2)  # (batch_size, seq_len, hidden_size * 2)
        scores = torch.tanh(self.attn(attn_input))  # (batch_size, seq_len, hidden_size)
        attn_weights = torch.softmax(scores @ self.v, dim=1)  # (batch_size, seq_len)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_size)
        return context.squeeze(1), attn_weights  # (batch_size, hidden_size), (batch_size, seq_len)


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


# Decoder with Attention
class Decoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, num_layers: int):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size + output_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x shape: (batch_size, 1, output_size)
        context, _ = self.attention(hidden[-1:], encoder_outputs)  # Compute attention
        lstm_input = torch.cat((x, context.unsqueeze(1)), dim=2)  # Concatenate input with context

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell


# Seq2Seq Model with Attention
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, target_length: int, teacher_forcing_ratio: float = 0.5, target=None):
        batch_size = src.shape[0]
        encoder_outputs, hidden, cell = self.encoder(src)

        decoder_input = torch.zeros(batch_size, 1, 1, device=src.device)  # [batch, 1, 1]

        outputs = []
        for t in range(target_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(output)
            # Teacher forcing: use actual target as next input with probability teacher_forcing_ratio
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t + 1, :]  # shape: (batch_size, 1, output_size)
            else:
                decoder_input = output  # use the prediction as next input

        # Concatenate predictions along the time dimension
        outputs = torch.cat(outputs, dim=1)  # shape: (batch_size, target_length, output_size)
        return outputs