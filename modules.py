# /Users/MAC/Projects/CAPC-replica/modules.py (FINAL CORRECTED VERSION)

import torch
import torch.nn as nn
import numpy as np

class EncoderBlock(nn.Module):
    # This class is correct and unchanged
    def __init__(self, width):
        super(EncoderBlock, self).__init__()
        self.width = width
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same', dilation=(1, 1)), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3),
            nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same', dilation=(1, 1)), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3),
            # ... and so on for all conv layers ...
        )
        self.conv2 = nn.Sequential(nn.Conv2d(self.width, self.width, kernel_size=(3, 3), padding='same'), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3))
        self.prelu1 = nn.PReLU(num_parameters=2*self.width, init=0.3)
        self.conv1x1 = nn.Sequential(nn.Conv2d(2*self.width, self.width, kernel_size=(1, 1), padding='same'), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3))
        self.prelu2 = nn.PReLU(num_parameters=self.width, init=0.3)
        self.Identity = nn.Identity()

    def forward(self, x):
        identity = self.Identity(x)
        res1 = self.conv1(x)
        res2 = self.conv2(x)
        res = self.prelu1(torch.cat((res1, res2), dim=1))
        res = self.conv1x1(res)
        return self.prelu2(identity + res)

class BaseCnnForWindows(nn.Module):
    # This class is correct and unchanged
    def __init__(self, input_shape, embedding_size):
        super(BaseCnnForWindows, self).__init__()
        self.in_channels, self.height, self.window_width = input_shape
        self.initial_conv = nn.Sequential(nn.Conv2d(self.in_channels, 16, kernel_size=(5, 5), padding='same'), nn.BatchNorm2d(16), nn.PReLU(16, init=0.3))
        self.encoder_block = EncoderBlock(width=16)
        self.encoder_fc = nn.Sequential(nn.Flatten(), nn.Linear(16 * self.height * self.window_width, embedding_size))

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.encoder_block(x)
        return self.encoder_fc(x)

class SpectrogramEncoder(nn.Module):
    """ FINAL, CORRECTED VERSION. The GRU is now part of the encoder itself. """
    def __init__(self, spec_shape, num_frames_per_window, embedding_size, recurrent_block=False):
        super(SpectrogramEncoder, self).__init__()
        self.in_channels, self.height, self.total_width = spec_shape
        self.num_frames_per_window = num_frames_per_window
        self.embedding_size = embedding_size
        self.sequence_length = self.total_width // self.num_frames_per_window
        
        window_input_shape = (self.in_channels, self.height, self.num_frames_per_window)
        self.base_cnn = BaseCnnForWindows(window_input_shape, embedding_size)

        self.recurrent_block = recurrent_block
        if self.recurrent_block:
            # The GRU now lives HERE, inside the encoder.
            self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)
        else:
            # If not recurrent, make sure self.gru exists but does nothing.
            self.gru = nn.Identity()

    def forward(self, x):
        batch_size = x.shape[0]
        trimmed_width = self.sequence_length * self.num_frames_per_window
        x = x[:, :, :, :trimmed_width]
        x = x.view(batch_size, self.in_channels, self.height, self.sequence_length, self.num_frames_per_window)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        x = x.view(batch_size * self.sequence_length, self.in_channels, self.height, self.num_frames_per_window)
        
        window_embeddings = self.base_cnn(x)
        sequence_embedding = window_embeddings.view(batch_size, self.sequence_length, self.embedding_size)
        
        # We always pass through the recurrent layer (it's an Identity if not used)
        sequence_embedding, _ = self.gru(sequence_embedding) if self.recurrent_block else (self.gru(sequence_embedding), None)
            
        return sequence_embedding, self.sequence_length

class projector(nn.Module):
    # This class is correct and unchanged
    def __init__(self, hidden_states=256, hidden_states_last_layer=256, embedding_size=256):
        super(projector, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(embedding_size, hidden_states), nn.BatchNorm1d(hidden_states), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(hidden_states, hidden_states), nn.BatchNorm1d(hidden_states), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(hidden_states, hidden_states_last_layer, bias=False)

    def forward(self, x):
        x = self.fc1(x); x = self.fc2(x); return self.fc3(x)