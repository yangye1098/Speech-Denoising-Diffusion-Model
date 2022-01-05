import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy import signal


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


class STFTEncoder():
    def __init__(self, N:int, L:int, stride:int, sample_rate=8000):
        """
        Inputs:
        N: the number of frequency bin
        L: the time window length
        stride: the time window step
        sample_rate: the sample rate of the audio input
        ... define any attributes you need
        """
        self.N = N
        self.L = L
        self.stride = stride
        self.sample_rate = sample_rate

    def encode(self, sound:torch.FloatTensor, condition_sound:torch.FloatTensor = None):

        """
        Encode sound using stft
        Inputs:
            sound: torch.FloatTensor of size [B, 1, T], B is the batch_size,
                T is the total time point of the sound,
                T is determined by L and stride :
                $T = (N - 1)/stride $
                so that it can produce N time bins after stft
            condition_sound: the noisy sound used to condition the model, same size as _sound_

        Output:
            real_gram: torch.FloatTensor of size [B, 1 or 2, N, N]
                carrying the real information after stft
            phase_gram: torch.FloatTensor of size [B, 1 or 2, N, N]
                carrying the imaginary information after stft

        """
        # stft of the sound
        # note there are B different sounds in the sound Tensor
        # each of them has the size of [1,1,T]
        # The argument stride along with L defines the noverlap in scipy.signal.stft
        # noverlap = stride - L

        if self.stride > self.L:
            noverlap = None
        else:
            noverlap = self.L - self.stride

        if condition_sound is None:
            # return the two output
            # the output from stft is complex number,
            # here is to get real and imagery part of the output
            # the size should both be [B, 1, N, N]
            # check out torch.unsqueeze to insert dimension
            f, t, Zxx = signal.stft(sound, self.sample_rate, nperseg=self.L, noverlap=noverlap)
            Zxx = Zxx[:, :, 1:, :]
            real = Zxx.real
            imaginary = Zxx.imag
            return torch.cat[torch.from_numpy(real), torch.from_numpy(imaginary), 1]
        else:
            # stft of condition_sound
            # concatenate two stft at dim = 1, check out torch.cat
            # return the two output
            # the size should both be [B, 2, N, N]
            _, _, Zxx = signal.stft(sound, self.sample_rate, nperseg=self.L, noverlap=noverlap)
            Zxx = Zxx[:, :, 1:, :]
            real_sound = torch.from_numpy(Zxx.real)
            imag_sound = torch.from_numpy(Zxx.imag)

            _, _, Zxx_condition = signal.stft(condition_sound, self.sample_rate, nperseg=self.L, noverlap=noverlap)
            Zxx_condition = Zxx_condition[:, :, 1:, :]
            real_con = torch.from_numpy(Zxx_condition.real)
            imag_con = torch.from_numpy(Zxx_condition.imag)

            real = torch.cat((real_sound, real_con), 1)
            imaginary = torch.cat((imag_sound, imag_con), 1)

            return real, imaginary


class STFTDecoder():
    def __init__(self, N:int, L:int, stride:int, sample_rate=8000):
        """
        Inputs:
        N: the number of frequency bin
        L: the time window length
        stride: the time window step
        sample_rate: the sample rate of the audio input
        ... define any attributes you need
        """
        self.N = N
        self.L = L
        self.stride = stride
        self.sample_rate = sample_rate

    def decode(self, real_gram:torch.FloatTensor, phase_gram:torch.FloatTensor):
        """
        Decode sound using istft, it is the reverse of the stft
        Inputs:
        real_gram: torch.FloatTensor of size [B, 1, N, N]
        carrying the real information after stft
        phase_gram: torch.FloatTensor of size [B, 1, N, N]
        carrying the imaginary information after stft

        Output:
        sound: torch.FloatTensor of [B, 1, T], B is the batch_size,
        T = (N-1)*stride

        """
        # istft of the input
        # return the reconstructed sound

        batch_size = real_gram.shape[0]
        Zxx = torch.zeros([batch_size, 1, self.N+1, self.N], dtype=torch.cfloat)
        ndim = real_gram.ndim
        z_temp = torch.cat([torch.unsqueeze(real_gram, ndim), torch.unsqueeze(phase_gram, ndim)], dim=ndim)
        Zxx[:, :, 1:, : ] = torch.view_as_complex(z_temp)
        _, sound = signal.istft(Zxx, self.sample_rate)
        sound = torch.from_numpy(sound)
        return sound


if __name__ == '__main__':
    from data import AudioDataset
    import torchaudio
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')

    import simpleaudio as sa
    from torch.utils.data import DataLoader

    sample_rate = 8000
    N = 128
    L = 256
    stride = 128
    T = (N - 1) * stride

    rng = np.random.default_rng()
    # generate or read test sound of size [1, 1, T]

    snr = 0
    dataroot = f'../data/data/wsj0_si_tr_{snr}'
    datatype = 'wav'
    dataset_tr = AudioDataset(dataroot, datatype, snr, T)
    dataset_tr.playIdx(2)

    dataloader = DataLoader(dataset_tr, batch_size=2)
    sound = next(iter(dataloader))

    # Construct a condition sound

    # dene encoder and decoder
    encoder = STFTEncoder(N, L, stride, sample_rate)
    decoder = STFTDecoder(N, L, stride, sample_rate)
    real, imaginary = encoder.encode(sound['Clean'], sound['Noisy'])
    # real, imaginary = encoder.encode(sound, x)  # with condition sound

#     assert(real.shape == torch.Size([1, 2, N, N]))
#     assert(imaginary.shape == torch.Size([1, 2, N, N]))

    # plot clean spectrogram
    plt.figure(1)
    z = torch.squeeze(real[0, 0, :, :]).numpy()**2 + torch.squeeze(imaginary[0, 0, :, :]).numpy()**2
    plt.pcolormesh(np.arange(128), np.arange(128), z)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)

    # plot the clean sound
    plt.figure(2)
    t = np.linspace(0, (T-1)/sample_rate, T)
    plt.plot(t, torch.squeeze(sound['Clean'][0, :, :]))
    plt.show(block=False)

    plt.show()
    # plot noisy spectrogram
    z = torch.squeeze(real[0, 0, :,  :]).numpy()**2 + torch.squeeze(imaginary[0, 0, :, :]).numpy()**2
    plt.pcolormesh(np.arange(128), np.arange(128), z)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show(block=False)
    plt.draw()

    # Another way to plot the spectrogram
    """
    sound_2 = torch.squeeze(sound).numpy()
    f, t, Sxx = spectrogram(sound_2, sample_rate, nperseg = L, noverlap=L-stride)
    plt.figure()
    plt.pcolormesh(t, f, Sxx)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    """

    # reconstruct back to sound
    rec_sound_clean = decoder.decode(real[:, 0, :, :], imaginary[:, 0, :, :])  # size [1, 1, T]
    rec_sound_noisy = decoder.decode(real[:, 1, :, :], imaginary[:, 1, :, :])  # size [1, 1, T]

    rec_sound_clean = torch.squeeze(rec_sound_clean)
    rec_sound_noisy = torch.squeeze(rec_sound_noisy)

    # listen to reconstructed sound
    play_obj = sa.play_buffer(torch.squeeze(sound['Clean']).numpy(), 1, 32//8, sample_rate)
    play_obj.wait_done()
    play_obj = sa.play_buffer(torch.squeeze(rec_sound_clean).numpy(), 1, 32//8, sample_rate)
    play_obj.wait_done()

    # plot the noisy sound
    t = np.linspace(0, (T-1)/sample_rate, T)
    plt.plot(t, rec_sound_noisy)
    plt.plot(t, torch.squeeze(sound['Noisy']))
    plt.show()

    # listen to reconstructed sound
    play_obj = sa.play_buffer(torch.squeeze(sound['Noisy']).numpy(), 1, 32//8, sample_rate)
    play_obj.wait_done()
    play_obj = sa.play_buffer(torch.squeeze(rec_sound_noisy).numpy(), 1, 32//8, sample_rate)
    play_obj.wait_done()
