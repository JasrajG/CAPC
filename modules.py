# modules.py
import torch
import torch.nn as nn
import numpy as np

class EncoderBlock(nn.Module):
    def __init__(self, width):
        super(EncoderBlock, self).__init__()
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

class BaseEncoder(nn.Module):
    def __init__(self, input_shape, embedding_size):
        super(BaseEncoder, self).__init__()
        self.input_size = np.prod(input_shape)
        self.width = input_shape[0]  # Channel dimension
        self.embedding_size = embedding_size
        self.encoder = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=(5, 5), padding='same'),
            nn.BatchNorm2d(self.width),
            nn.PReLU(num_parameters=self.width, init=0.3),
            EncoderBlock(self.width),
        )
        self.encoder_fc = nn.Sequential(nn.Flatten(), nn.Linear(self.input_size, embedding_size))

    def forward(self, x):
        x = self.encoder(x)
        return self.encoder_fc(x)

class RecurrentEncoder(BaseEncoder):
    def __init__(self, input_type, num_frames, embedding_size, recurrent_block=False):
        self.input_type = input_type
        self.num_frames = num_frames
        if input_type == 'SignFi':
            self.CHANNELS = 3
            self.SUBCARRIERS = 30
            self.TOTAL_TIME = 200
            self.input_shape = (self.CHANNELS, self.num_frames, self.SUBCARRIERS)
            self.sequence_length = self.TOTAL_TIME // self.num_frames
        else:
            raise NotImplementedError

        super(RecurrentEncoder, self).__init__(self.input_shape, embedding_size)
        self.recurrent_block = recurrent_block
        if self.recurrent_block:
            self.lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True)

    def forward(self, x, view_mode='in_sequence'):
        batch_size = x.shape[0]
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(batch_size, self.CHANNELS, self.sequence_length, self.num_frames, self.SUBCARRIERS)
        x = x.permute(0, 2, 1, 3, 4)
        new_x = x.contiguous().view(batch_size * self.sequence_length, self.CHANNELS, self.num_frames, self.SUBCARRIERS)
        new_x = super(RecurrentEncoder, self).forward(new_x)
        sequence_embedding = new_x.view(batch_size, self.sequence_length, self.embedding_size)

        if self.recurrent_block:
            sequence_embedding, _ = self.lstm(sequence_embedding)

        if view_mode == 'flat':
            return sequence_embedding.view(batch_size, -1), self.embedding_size * self.sequence_length
        elif view_mode == 'in_batch':
            return sequence_embedding.view(-1, self.embedding_size), batch_size
        else:
            return sequence_embedding, self.sequence_length

class projector(nn.Module):
    def __init__(self, hidden_states=256, hidden_states_last_layer=256, embedding_size=256):
        super(projector, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(embedding_size, hidden_states), nn.BatchNorm1d(hidden_states), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(hidden_states, hidden_states), nn.BatchNorm1d(hidden_states), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(hidden_states, hidden_states_last_layer, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)