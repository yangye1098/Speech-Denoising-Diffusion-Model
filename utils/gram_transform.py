from torchaudio import transforms as TT
from torch import nn
import torch

def gram_log_modulus_normalize(gram, expand_order):
    # order is used to spread values apart, when most value is concentrated at (10e-4, 10e-2)
    # -1 < grams < 1
    # log_modulus = sign(x) * log10(|x * 10**transform_order| + 1)
    # -order < log_modulus < order
    # normalize =  log_modulus + order / (2*order)
    gram_log_modulus = torch.sign(gram)*torch.log10(torch.abs(10.**expand_order *gram) + 1.)
    normalize_order = expand_order+2
    return (gram_log_modulus+normalize_order)/(2.*(normalize_order))

def gram_log_modulus_normalize_reverse(gram_norm, expand_order):
    # reverse normalization
    normalize_order = expand_order+2
    gram_log_modulus = gram_norm * 2. * (normalize_order) - (normalize_order)
    sign = torch.sign(gram_log_modulus)
    return sign * (torch.pow(10, torch.abs(gram_log_modulus)) - 1) / 10.**expand_order


class RISpectrograms(nn.Module):
    def __init__(self, N:int, L:int, stride:int, sample_rate, expand_order=4, use_mel=True, normalize=True):
        super().__init__()
        # power is None, return the original complex spectrum
        # normalized is False, we'll normalize later ourselves
        self.spectrogram_RI = TT.Spectrogram(n_fft=L,
                                        hop_length=stride,
                                        power=None,
                                        normalized=False,
                                        onesided=True,
                                        return_complex=True)
        if use_mel:
            self.mel_scale = TT.MelScale(n_mels=N,
                                    sample_rate=sample_rate,
                                    n_stft=L//2+1,
                                    norm=None,
                                    mel_scale='htk')

        self.normalize = normalize
        self.use_mel = use_mel
        self.expand_order = expand_order


    def forward(self, audio:torch.FloatTensor):

        """
        audio: tensor of size [T]
        """
        spectrogram_complex = self.spectrogram_RI(audio)

        # switch the dimensions
        # split to two channels
        RI_temp = torch.view_as_real(spectrogram_complex)
        # switch the dimensions
        RI_temp = torch.movedim(RI_temp, -1, -3).contiguous()

        # map to mel scale
        if self.use_mel:
            RI_grams = self.mel_scale(RI_temp)
        else:
            # ignore the first bin
            RI_grams = RI_temp

        if self.normalize:
            return gram_log_modulus_normalize(RI_grams, self.expand_order)
        else:
            return RI_grams



class ReverseRISpectrograms(nn.Module):

    def __init__(self, N:int, L:int, stride:int, sample_rate, expand_order=4, use_mel=False, normalize=True):
        super().__init__()


        if use_mel:
            self.inverse_mel = TT.InverseMelScale(n_stft=L//2+1,
                                             n_mels=N,
                                             sample_rate=sample_rate,
                                             norm=None,
                                             mel_scale='htk')

        self.inverse_spectrogram = TT.InverseSpectrogram(n_fft=L,
                                                         hop_length=stride,
                                                         normalized=False,
                                                         onesided=True)
        self.use_mel = use_mel
        self.expand_order = expand_order
        self.normalize = normalize


    def forward(self, RI_grams:torch.FloatTensor):
        """
        RI_grams: tensor of size [2, n_fft, n_time]
        """
        if self.normalize:
            RI_grams = gram_log_modulus_normalize_reverse(RI_grams, self.expand_order)

        if self.use_mel:
            RI_temps = self.inverse_mel(RI_grams)
        else:
            # get back the removed frequency bin
            RI_temps = RI_grams

        RI_temps = torch.movedim(RI_temps, -3, -1)
        spectrogram = torch.view_as_complex(RI_temps.contiguous())
        return self.inverse_spectrogram(spectrogram)

