import torch
from torch import nn as nn

# Encoder: Processes the input sequence and returns the final hidden and cell states.
class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


# Decoder: Uses the encoder's states to generate the future sequence.
class Decoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, num_layers: int):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size, 1, output_size)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        # prediction shape: (batch_size, 1, output_size)
        return prediction, hidden, cell


# Seq2Seq Model: Combines the encoder and decoder.
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, target_length: int, teacher_forcing_ratio: float = 0.5, target=None):
        batch_size = src.shape[0]
        # Encode the input sequence
        hidden, cell = self.encoder(src)

        decoder_input = torch.zeros(batch_size, 1, 1, device=src.device)  # [batch, 1, 1]

        outputs = []
        for t in range(target_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(output)
            # Teacher forcing: use actual target as next input with probability teacher_forcing_ratio
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t + 1, :]  # shape: (batch_size, 1, output_size)
            else:
                decoder_input = output  # use the prediction as next input

        # Concatenate predictions along the time dimension
        outputs = torch.cat(outputs, dim=1)  # shape: (batch_size, target_length, output_size)
        return outputs