import torch
from torch import nn as nn


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, forecast_seq_len, num_layers=1):
        """
        Args:
            input_size (int): Number of features in the input.
            hidden_size (int): Hidden state size.
            output_size (int): Number of output features (1 if predicting GHI).
            forecast_seq_len (int): Length of the forecast sequence.
            num_layers (int): Number of LSTM layers.
        """
        super(LSTMEncoderDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.forecast_seq_len = forecast_seq_len

        # Encoder LSTM that processes the input sequence
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Decoder LSTM that generates the forecast one step at a time.
        # It expects an input size of 1 because we feed back the previous output.
        self.decoder = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        # A fully connected layer to map the LSTM output to the target value.
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        """
        Args:
            x (Tensor): Input tensor of shape [batch, input_seq_len, features].
            target (Tensor, optional): Ground truth tensor of shape [batch, forecast_seq_len, 1].
            teacher_forcing_ratio (float): Probability of using teacher forcing.
        Returns:
            outputs (Tensor): Predictions of shape [batch, forecast_seq_len, output_size].
        """
        batch_size = x.size(0)
        # Encode the input sequence
        _, (hidden, cell) = self.encoder(x)  # hidden: (num_layers, batch, hidden_size)

        # Initialize the decoder input (using zeros)
        decoder_input = torch.zeros(batch_size, 1, 1, device=x.device)  # [batch, 1, 1]
        outputs = []

        for t in range(self.forecast_seq_len):
            # Pass through decoder LSTM
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            # Map the output to the prediction using a linear layer
            out = self.fc(out)  # [batch, 1, output_size]
            outputs.append(out)

            # Decide whether to use teacher forcing
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Use the ground truth value as the next input
                # target[:, t] is of shape [batch, 1] so we unsqueeze to get [batch, 1, 1]
                decoder_input = target[:, t].unsqueeze(1)
            else:
                # Use the prediction as the next input
                decoder_input = out

        # Concatenate all time steps to form the full sequence output
        outputs = torch.cat(outputs, dim=1)  # [batch, forecast_seq_len, output_size]
        return outputs


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

        # For the first decoder input, we use the last value of the source sequence.
        # decoder_input = src[:, -1:, :]  # shape: (batch_size, 1, input_size)
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