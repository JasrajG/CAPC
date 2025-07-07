# /Users/MAC/Projects/CAPC-replica/modules.py

import torch
import torch.nn as nn
import numpy as np

class EncoderBlock(nn.Module):
    """
    This is the core Separable-Convolutional block from RSCNet.
    This part is perfect as a generic CNN feature extractor for our windows.
    NO CHANGES NEEDED HERE.
    """
    def __init__(self, width):
        super(EncoderBlock, self).__init__()
        # ... (The code for EncoderBlock remains identical to your original file)
        self.width = width
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same', dilation=(1, 1)), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3),
            nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same', dilation=(1, 1)), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3),
            nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same', dilation=(2, 2)), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3),
            nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same', dilation=(2, 2)), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3),
            nn.Conv2d(self.width, self.width, kernel_size=(3, 1), padding='same', dilation=(3, 3)), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3),
            nn.Conv2d(self.width, self.width, kernel_size=(1, 3), padding='same', dilation=(3, 3)), nn.BatchNorm2d(self.width), nn.PReLU(num_parameters=self.width, init=0.3),
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
    """
    This is the simple CNN that will process ONE window of the spectrogram.
    It's the equivalent of the original paper's "Separable-Convolutional Block".
    """
    def __init__(self, input_shape, embedding_size):
        super(BaseCnnForWindows, self).__init__()
        # input_shape is (channels, height, width_of_one_window)
        self.in_channels = input_shape[0]
        self.height = input_shape[1]
        self.window_width = input_shape[2]

        # Initial convolution to get to the desired 'width' for the EncoderBlock
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=(5, 5), padding='same'),
            nn.BatchNorm2d(16),
            nn.PReLU(16, init=0.3)
        )
        self.encoder_block = EncoderBlock(width=16)
        
        # Final layers to produce the embedding for this single window
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * self.height * self.window_width, embedding_size)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.encoder_block(x)
        embedding = self.encoder_fc(x)
        return embedding

class SpectrogramEncoder(nn.Module):
    """
    This is the new, FAITHFUL replacement for the original RecurrentEncoder.
    It performs the correct windowing logic for spectrograms.
    """
    def __init__(self, spec_shape, num_frames_per_window, embedding_size, recurrent_block=False):
        super(SpectrogramEncoder, self).__init__()
        # spec_shape is (channels, height, total_width) e.g., (1, 129, 184)
        self.in_channels = spec_shape[0]
        self.height = spec_shape[1]
        self.total_width = spec_shape[2]
        self.num_frames_per_window = num_frames_per_window # The width of each small slice
        self.embedding_size = embedding_size

        # Calculate how many windows we can create from the spectrogram
        self.sequence_length = self.total_width // self.num_frames_per_window
        
        # Define the shape of a SINGLE window that the CNN will see
        window_input_shape = (self.in_channels, self.height, self.num_frames_per_window)

        # The CNN that processes each window
        self.base_cnn = BaseCnnForWindows(window_input_shape, embedding_size)

        # The recurrent part, identical to the original paper's concept
        self.recurrent_block = recurrent_block
        if self.recurrent_block:
            self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)

    def forward(self, x):
        # x is the full spectrogram: (batch, channels, height, total_width)
        batch_size = x.shape[0]

        # --- This is the crucial windowing step ---
        # 1. Trim the spectrogram so its width is perfectly divisible by our window size
        trimmed_width = self.sequence_length * self.num_frames_per_window
        x = x[:, :, :, :trimmed_width]
        
        # 2. Reshape to create the sequence of windows
        # Shape becomes: (batch, channels, height, num_windows, window_width)
        x = x.view(batch_size, self.in_channels, self.height, self.sequence_length, self.num_frames_per_window)
        
        # 3. Permute and flatten to create a big batch of windows for the CNN
        # (batch * num_windows, channels, height, window_width)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        x = x.view(batch_size * self.sequence_length, self.in_channels, self.height, self.num_frames_per_window)
        
        # --- Pass each window through the CNN ---
        # Output shape: (batch * num_windows, embedding_size)
        window_embeddings = self.base_cnn(x)
        
        # --- Reshape back into a sequence ---
        # Output shape: (batch, num_windows, embedding_size)
        sequence_embedding = window_embeddings.view(batch_size, self.sequence_length, self.embedding_size)
        
        # --- Pass the sequence through the recurrent model (if enabled) ---
        if self.recurrent_block:
            sequence_embedding, _ = self.gru(sequence_embedding)
            
        return sequence_embedding, self.sequence_length


class projector(nn.Module):
    """
    This projector is used by the SSL models (BarlowTwins, etc.).
    Kept as is. NO CHANGES NEEDED HERE.
    """
    def __init__(self, hidden_states=256, hidden_states_last_layer=256, embedding_size=256):
        super(projector, self).__init__()
        # ... (The code for projector remains identical to your original file)
        self.fc1 = nn.Sequential(nn.Linear(embedding_size, hidden_states), nn.BatchNorm1d(hidden_states), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(hidden_states, hidden_states), nn.BatchNorm1d(hidden_states), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(hidden_states, hidden_states_last_layer, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)