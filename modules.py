# modules.py (UPGRADED for full AutoFi implementation)

import torch
import torch.nn as nn
import numpy as np

# --- The EncoderBlock1D, BaseCnnForIQWindows, and IQEncoder classes are UNCHANGED ---
# (I am including them here so you can do a full copy-paste)
class EncoderBlock1D(nn.Module):
    def __init__(self, width):
        super(EncoderBlock1D, self).__init__(); self.width = width
        self.conv1 = nn.Sequential(nn.Conv1d(self.width, self.width, kernel_size=11, padding='same'), nn.BatchNorm1d(self.width), nn.PReLU(num_parameters=self.width, init=0.3))
        self.conv2 = nn.Sequential(nn.Conv1d(self.width, self.width, kernel_size=5, padding='same'), nn.BatchNorm1d(self.width), nn.PReLU(num_parameters=self.width, init=0.3))
        self.prelu1 = nn.PReLU(num_parameters=2*self.width, init=0.3); self.conv1x1 = nn.Sequential(nn.Conv1d(2*self.width, self.width, kernel_size=1, padding='same'), nn.BatchNorm1d(self.width), nn.PReLU(num_parameters=self.width, init=0.3)); self.prelu2 = nn.PReLU(num_parameters=self.width, init=0.3); self.Identity = nn.Identity()
    def forward(self, x):
        identity = self.Identity(x); res1 = self.conv1(x); res2 = self.conv2(x); res = self.prelu1(torch.cat((res1, res2), dim=1)); res = self.conv1x1(res); return self.prelu2(identity + res)

class BaseCnnForIQWindows(nn.Module):
    def __init__(self, input_shape, embedding_size):
        super(BaseCnnForIQWindows, self).__init__(); self.in_channels, self.window_length = input_shape
        self.initial_conv = nn.Sequential(nn.Conv1d(self.in_channels, 16, kernel_size=11, padding='same'), nn.BatchNorm1d(16), nn.PReLU(16, init=0.3))
        self.encoder_block = EncoderBlock1D(width=16)
        self.encoder_fc = nn.Sequential(nn.Flatten(), nn.Linear(16 * self.window_length, embedding_size))
    def forward(self, x): x = self.initial_conv(x); x = self.encoder_block(x); return self.encoder_fc(x)

class IQEncoder(nn.Module):
    def __init__(self, iq_shape, num_samples_per_window, embedding_size, recurrent_block=False):
        super(IQEncoder, self).__init__(); self.in_channels, self.total_length = iq_shape; self.num_samples_per_window = num_samples_per_window; self.embedding_size = embedding_size; self.sequence_length = self.total_length // self.num_samples_per_window
        self.base_cnn = BaseCnnForIQWindows((self.in_channels, self.num_samples_per_window), embedding_size)
        self.recurrent_block = recurrent_block
        if self.recurrent_block: self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)
        else: self.gru = nn.Identity()
    def forward(self, x):
        batch_size = x.shape[0]; trimmed_length = self.sequence_length * self.num_samples_per_window; x = x[:, :, :trimmed_length]
        x = x.view(batch_size, self.in_channels, self.sequence_length, self.num_samples_per_window).permute(0, 2, 1, 3).contiguous().view(batch_size * self.sequence_length, self.in_channels, self.num_samples_per_window)
        window_embeddings = self.base_cnn(x); sequence_embedding = window_embeddings.view(batch_size, self.sequence_length, self.embedding_size)
        sequence_embedding, _ = self.gru(sequence_embedding) if self.recurrent_block else (self.gru(sequence_embedding), None)
        return sequence_embedding, self.sequence_length

# --- THIS IS THE FIRST NEW PIECE: The Main Projector ---
class Projector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size, bias=False)
        )
    def forward(self, x):
        return self.net(x)

# --- THIS IS THE SECOND NEW PIECE: The Symbol Prediction Head ---
# This is a direct implementation of the architecture described in Section III-A of the AutoFi paper.
class SymbolPredictionHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_symbols):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_symbols)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x starts as (batch_size, flat_embedding_size)
        # We need to add a sequence dimension for the GRU: (batch_size, 1, flat_embedding_size)
        x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        # We take the output of the GRU and remove the sequence dimension
        gru_out = gru_out.squeeze(1)
        logits = self.fc(gru_out)
        return self.softmax(logits) # Return the "soft label" probabilities