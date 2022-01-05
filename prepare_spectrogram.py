
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

    if stride > L:
        noverlap = None
    else:
        noverlap = L - stride

    f, t, Zxx = signal.stft(sound, sample_rate, nperseg=L, noverlap=noverlap)

    Zxx = Zxx[:,  1:, :]

    # concatenate two gram together
    grams = torch.cat([torch.FloatTensor(Zxx.real), torch.FloatTensor(Zxx.imag)], 0)
    # save
    # The saved array should have size of [2, N, N]
    # convert logscale in [0, 10]
    grams_logscale = torch.log10(torch.clamp(grams, min=1e-10)) + 10
    # scale to [0,1]
    grams_logscale = grams_logscale * (1-0)/(10 - 0)
    np.save(f'{filename}.spec.npy', grams_logscale.cpu().numpy())


def main(dir, N, L, stride, T, sample_rate=8000):
    filenames = glob(f'{dir}/**/*.wav', recursive=True)
    transform_dir = partial(transform, N=N, L=L, stride=stride, T=T, sample_rate=sample_rate)

    # multiprocess processing
    with ProcessPoolExecutor() as executor:
      list(tqdm(executor.map(transform_dir, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
    dir = './data/data/wsj0_si_val_0'
    N = 128
    L = 256
    stride = 128
    T = (N-1) * stride
    sample_rate = 8000

    main(dir, N, L, stride, T, sample_rate)
