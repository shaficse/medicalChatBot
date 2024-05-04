import torch
from torch import nn
import torch.nn.functional as F

class Attn(nn.Module):
    """Attention Module for Luong's attention mechanism."""
    def __init__(self, method, hidden_size):
        """
        Args:
            method (str): The type of attention ('dot', 'general', 'concat').
            hidden_size (int): The size of the hidden state.
        """
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(f"{self.method} is not an appropriate attention method.")
        self.hidden_size = hidden_size

        # Define layers according to the attention method
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        """Compute the dot product score for the 'dot' attention mechanism."""
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        """Compute the general score for the 'general' attention mechanism."""
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        """Compute the concat score for the 'concat' attention mechanism."""
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        """Calculate the attention weights (energies) based on the given method."""
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()  # Transpose max_length and batch_size dimensions
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # Return the softmax normalized probability scores

class EncoderRNN(nn.Module):
    """Encoder RNN that processes text input to hidden states."""
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        """
        Args:
            hidden_size (int): The size of the hidden state.
            embedding (nn.Embedding): The embedding layer.
            n_layers (int, optional): Number of layers in the GRU.
            dropout (float, optional): Dropout rate.
        """
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """Forward pass through the encoder returning output and hidden states."""
        embedded = self.embedding(input_seq)  # Convert word indexes to embeddings
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)  # Pack padded batch of sequences
        outputs, hidden = self.gru(packed, hidden)  # Forward pass through GRU
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)  # Unpack padding
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden

class LuongAttnDecoderRNN(nn.Module):
    """Decoder RNN that generates the next word in the sequence."""
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        """
        Args:
            attn_model (str): The type of attention mechanism.
            embedding (nn.Embedding): Embedding layer.
            hidden_size (int): The size of the hidden state.
            output_size (int): The size of the output vocabulary.
            n_layers (int, optional): Number of layers in the GRU.
            dropout (float, optional): Dropout rate.
        """
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """Forward pass through the decoder."""
        embedded = self.embedding(input_step)  # Get embedding of current input word
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)  # Forward through unidirectional GRU
        attn_weights = self.attn(rnn_output, encoder_outputs)  # Calculate attention weights
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # Multiply attention weights to encoder outputs
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden