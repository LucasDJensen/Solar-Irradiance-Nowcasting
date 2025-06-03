import torch
from torch import nn

# Define the simple LSTM model from earlier
class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, dropout: float = 0.5):
        """
        A simple LSTM network that processes the entire input sequence and
        applies a fully connected layer at every time step to produce the output.

        Args:
            input_size (int): Number of expected features in the input.
            hidden_size (int): Number of features in the hidden state.
            output_size (int): Number of features in the output.
            num_layers (int): Number of recurrent layers.
        """
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Tensor: Output predictions for each time step with shape (batch_size, seq_len, output_size)
        """
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out = torch.relu(lstm_out)
        output = self.fc(lstm_out)  # output shape: (batch_size, seq_len, output_size)
        return output
