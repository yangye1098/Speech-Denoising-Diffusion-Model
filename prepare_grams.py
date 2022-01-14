import argparse
import utils.config as Config
import numpy as np
import torch
import torchaudio

from glob import glob
from tqdm import tqdm

from utils.gram_transform import RISpectrograms


def main(dir, encoder_type, datatype, N, L, stride, T, sample_rate=8000, expand_order=5, normalize=True, debug=False):
    filenames = glob(f'{dir}/**/*.wav', recursive=True)

    if encoder_type == 'RI_mel':
        transform = RISpectrograms(N, L, stride, sample_rate, expand_order=expand_order, use_mel=True, normalize=normalize )
    elif encoder_type == 'RI':
        transform = RISpectrograms(N, L, stride, sample_rate, expand_order=expand_order, use_mel=False, normalize=normalize)

    else:
        raise NotImplementedError

    def worker(filename, debug=False):
        sound, sr = torchaudio.load(filename, num_frames=T)
        assert sr == sample_rate
        if debug:

            grams = transform(torch.squeeze(sound))
            plotTwoGrams(grams)
            plotSpectrogram(grams, expand_order)
            plt.show()
            play_audio(sound, '.wav', sample_rate, N, L, stride, expand_order, normalize=True)
            play_audio(grams, datatype, sample_rate, N, L, stride, expand_order, normalize=True)

        else:
            grams_normalized = transform(torch.squeeze(sound))
            if grams_normalized.min() < 0 or grams_normalized.max() > 1:
                raise ValueError(f'Out of Range: min:{grams_normalized.min()}, max:{grams_normalized.max()}')

            np.save(f'{filename}{datatype}', grams_normalized.cpu().numpy())

    for i, f in tqdm(enumerate(filenames), desc='Preprocessing', total=len(filenames)):
        worker(f, debug)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.gram_transform import gram_log_modulus_normalize_reverse
    from utils.gram_transform import ReverseRISpectrograms

    try:
        import simpleaudio as sa
        hasAudio = True
    except ModuleNotFoundError:
        hasAudio = False

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        help='the directory containing wav files to be prepared')
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-n', '--normalize', type=bool, default=True)
    parser.add_argument('-debug', '-d', action='store_true')

    args = parser.parse_args()


    # parse argument
    dir = args.path
    opt = Config.parse(args)

    sample_rate = opt['sample_rate']

    encoder_opt = opt['model']['encoder']
    encoder_type = opt['model']['encoder']['type']
    datatype = opt['datatype']
    N = encoder_opt[encoder_type]['N']
    L = encoder_opt[encoder_type]['L']
    stride =encoder_opt[encoder_type]['stride']
    expand_order = encoder_opt[encoder_type]['expand_order']

    T = (N-1) * stride


    def play_audio(data: torch.FloatTensor, datatype, sample_rate, N=None, L=None, stride=None, expand_order=None, normalize=True):
        if hasAudio:
            if datatype == '.wav':
                sound = data.numpy()

            elif datatype == '.mel.npy':
                transform = ReverseRISpectrograms(N=N, L=L, stride=stride, sample_rate=sample_rate,
                                                  expand_order=expand_order, use_mel=True, normalize=normalize)
                sound = torch.squeeze(transform(data)).numpy()
            elif datatype == '.spec.npy':
                transform = ReverseRISpectrograms(N=N, L=L, stride=stride, sample_rate=sample_rate,
                                                  expand_order=expand_order, use_mel=False, normalize=normalize)
                sound = torch.squeeze(transform(data)).numpy()

            play_obj = sa.play_buffer(sound, 1, 32 // 8, sample_rate)
            play_obj.wait_done()


    def plotSpectrogram(grams, expand_order):

        grams_original = gram_log_modulus_normalize_reverse(grams, expand_order)
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

    print(f'.wav file sample rate is {sample_rate}')
    print(f'prepare {encoder_type} in {datatype}: N:{N}, L:{L}, stride:{stride}, expand_order:{expand_order}, normalize:{args.normalize}')
    main(dir, encoder_type, datatype, N, L, stride, T, sample_rate, expand_order, args.normalize, args.debug)
