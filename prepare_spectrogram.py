
import numpy as np
import torch
import torchaudio

from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm
from functools import partial
from scipy import signal

def gram_log_modulus_normalize(gram, n):
    return (np.sign(gram)*np.log10(np.abs((10**n)*gram)+1)+n)/(2*n)

def gram_log_modulus_normalize_revers(gram_norm, n):
    gram_log_modulus = gram_norm * 2 * n - n
    sign = np.sign(gram_log_modulus)
    return sign * (10. ** (sign * gram_log_modulus) - 1) / 10. ** n




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

    if grams.min() < -1:
        raise ValueError('minimum value is less than -1')

    # save
    # The saved array should have size of [2, N, N]
    # convert logscale in [0, 10]
    # grams_logscale = torch.log10(torch.clamp(grams, min=1e-10)) + 10

    #plt.hist(grams[0, :,: ].view(1,-1)*1000)

    log_modulus = lambda x, n: (np.sign(x)*np.log10(np.abs((10**n)*x)+1)+n)/(2*n)
    #plt.figure()
    #plt.hist(log_modulus(grams[0, :,: ].view(1,-1), 3))
    #plt.figure()
    #plt.hist(np.log10(np.abs(grams[0, :,: ].view(1,-1))))
    grams_logscale = log_modulus(grams, 3)
    plotTwoGrams(grams_logscale)
    plotSpectrogram(grams_logscale, 3)

    plotTwoGrams(np.log10(np.abs(grams)))
    # scale to [0,1]
    #grams_logscale = grams_logscale * (1-0)/(10 - 0)


    # np.save(f'{filename}.spec.npy', grams_logscale.cpu().numpy())


def main(dir, N, L, stride, T, sample_rate=8000):
    filenames = glob(f'{dir}/**/*.wav', recursive=True)
    transform_dir = partial(transform, N=N, L=L, stride=stride, T=T, sample_rate=sample_rate)

    for i, f in enumerate(filenames):
        transform_dir(f)
        plt.show()
        print(i)

    # multiprocess processing
    # with ProcessPoolExecutor() as executor:
    #   list(tqdm(executor.map(transform_dir, filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dir = './data/data/wsj0_si_tr_0/clean'
    N = 128
    L = 256
    stride = 128
    T = (N-1) * stride
    sample_rate = 8000


    def plotSpectrogram(grams, n):

        def reverse_log_modulus(x_norm):
            x = x_norm * 2*n - n
            sign = np.sign(x)
            return sign * (10.**(sign*x ) - 1) / 10.**n


        grams_original = reverse_log_modulus(grams)
        spectrogram = np.sum([np.square(grams_original[0, :, :]), np.square(grams_original[1, :, :])])

        plt.figure()
        im = plt.imshow(20*np.log10(spectrogram), origin='lower', aspect='auto')
        plt.colorbar(im)
        plt.xlabel('Frame')
        plt.ylabel('Frequency')
        plt.title('Spectrogram, db re 1')
        plt.show(block=False)

    def plotTwoGrams(grams):
        fig, axs = plt.subplots(1, 2)
        for i in [0, 1]:
            axs[i].set_ylabel('frequency')
            axs[i].set_xlabel('frame')
            im = axs[i].imshow(grams[i,:,:], origin='lower', aspect='auto')
            fig.colorbar(im, ax=axs[i])

        axs[0].set_title('Real')
        axs[1].set_title('Imaginary')
        plt.show(block=False)


    main(dir, N, L, stride, T, sample_rate)
