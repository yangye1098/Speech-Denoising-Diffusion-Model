import torch.nn as nn
import torch
import torch.nn.functional as F

# encode sound to 2d representation using 1d convolution
class ConvEncoder(nn.Module):
    def __init__(self, N, L, stride):
        super(ConvEncoder, self).__init__()
        """
            N: the encoding dimension
            L: the segment Length
            stride: the stride size to cut the segments
        """

        self.L = L
        self.N = N
        self.stride = stride
        # output from conv1d_U and conv1d_V is [B, N, K],
        # K is determined by mixture length, L and stride
        # K = floor( (T-L)/stride + 1), T is the total time points of the mixture

        self.U = nn.Conv1d(1, self.N, kernel_size=self.L, stride=self.stride, bias=False)
        self.V = nn.Conv1d(1, self.N, kernel_size=self.L, stride=self.stride, bias=False)

    def forward(self, sound, condition_snd=None):
        """
        Args:
            sound: the noisy sound [B, 1 , T], T is the number of time points.
            sound: the reference sound [B, 1 , T], T is the number of time points.
        Returns:
            sound_representation: [B, 1 or 2, N, K], K is determined by mixture length, L and stride K = floor( (T-L)/stride + 1)
        """

        sound_representation = torch.unsqueeze(F.relu(self.U(sound))*torch.sigmoid(self.V(sound)), 1) # [B, 1, N, K]
        if condition_snd is not None:
            condition_representation = torch.unsqueeze(F.relu(self.U(condition_snd)) * torch.sigmoid(self.V(condition_snd)), 1)  # [B, 1, N, K]
            # just concatenate
            return torch.cat([sound_representation, condition_representation], dim=1)
        else:
            return sound_representation


# decode sound from 2d representation
class ConvDecoder(nn.Module):
    def __init__(self, N, L, stride):
        super(ConvDecoder, self).__init__()
        self.N = N
        self.L = L
        self.stride = stride
        # The transposed convolution
        self.transposeConv = nn.ConvTranspose1d(self.N, 1, self.L, bias=False, stride=self.stride)


    def forward(self, output_representation):
        """
        Args:
            output_representation: [B, 1, N, K]
        Return:
            denoised_snd: [B, 1, 1, T], T is the total time point of the signal
        """
        # Prepare mask for decoding
        denoised_snd = self.transposeConv(torch.squeeze(output_representation)) #[B, 1, T]
        return denoised_snd

