
import numpy as np
import torch
import torchaudio


from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm
from functools import partial
from scipy import signal



def transform(filename, N:int, L:int, stride:int, T:int, sample_rate=8000):

    sound, sr = torchaudio.load(filename, num_frames=T)
    assert sr==sample_rate

    # calculate the two grams
    # this time the audio has size of [1, T]

    # calculate real and phase gram in size of [1, N, N]

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=L,
        hop_length=stride,
        power=2.0,
        n_mels=N,
        f_max=sr/2.0,
        f_min=20.0,
        normalized=True
    )


    with torch.no_grad():
        spectrogram = mel_spectrogram(sound)
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)

        np.save(f'{filename}.mel.npy', spectrogram.cpu().numpy())



def main(dir, N, L, stride, T, sample_rate=8000):
    filenames = glob(f'{dir}/**/*.wav', recursive=True)
    transform_dir = partial(transform, N=N, L=L, stride=stride, T=T, sample_rate=sample_rate)

    # multiprocess processing
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(transform_dir, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
    dir = './data/data/wsj0_si_tr_0'
    N = 128
    L = 512
    stride = 256
    T = (N-1) * stride
    sample_rate = 8000

    main(dir, N, L, stride, T, sample_rate)
