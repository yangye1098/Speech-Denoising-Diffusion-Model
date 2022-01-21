from pathlib import Path
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset



def generate_inventory(path, file_type='.wav'):
    path = Path(path)
    assert path.is_dir(), '{:s} is not a valid directory'.format(path)

    file_paths = path.glob('*'+file_type)
    file_names = [ file_path.name for file_path in file_paths ]
    assert file_names, '{:s} has no valid {} file'.format(path, file_type)
    return file_names


class AudioDataset(Dataset):
    def __init__(self, dataroot, datatype, sample_rate=8000, T=-1):
        if datatype not in ['.wav', '.spec.npy', '.mel.npy']:
            raise NotImplementedError
        self.datatype = datatype
        self.sample_rate = sample_rate
        self.T = T

        self.clean_path = Path('{}/clean'.format(dataroot))
        self.noisy_path = Path('{}/noisy'.format(dataroot))

        self.inventory = generate_inventory(self.clean_path, datatype)
        self.data_len = len(self.inventory)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if self.datatype == '.wav':
            clean, sr = torchaudio.load(self.clean_path/self.inventory[index], num_frames=self.T)
            assert(sr==self.sample_rate)
            noisy, sr = torchaudio.load(self.noisy_path/self.inventory[index], num_frames=self.T)
            assert (sr == self.sample_rate)
        elif self.datatype == '.spec.npy' or self.datatype == '.mel.npy':
            # load the two grams
            clean = torch.from_numpy(np.load(self.clean_path/self.inventory[index]))
            noisy = torch.from_numpy(np.load(self.noisy_path/self.inventory[index]))

        return clean, noisy, index

    def getName(self, idx):
        filename_noext = self.inventory[idx].lsplit('.', 1)[0]
        return filename_noext




class OutputDataset(AudioDataset):
    def __init__(self, dataroot, datatype, sample_rate=8000, T=-1):
        super().__init__(dataroot, datatype, sample_rate, T)

        self.noisy_path = Path('{}/noisy'.format(dataroot))
        self.output_path = Path('{}/output'.format(dataroot))
        self.inventory = sorted(self.inventory)


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if self.datatype == '.wav':
            clean, sr = torchaudio.load(self.clean_path/self.inventory[index], num_frames=self.T)
            assert(sr==self.sample_rate)
            noisy, sr = torchaudio.load(self.noisy_path/self.inventory[index], num_frames=self.T)
            assert (sr == self.sample_rate)
            output, sr = torchaudio.load(self.output_path/self.inventory[index], num_frames=self.T)
            assert (sr == self.sample_rate)
        elif self.datatype == '.spec.npy' or self.datatype == '.mel.npy':
            # load the two grams
            clean = torch.from_numpy(np.load(self.clean_path/self.inventory[index]))
            noisy = torch.from_numpy(np.load(self.noisy_path/self.inventory[index]))
            output = torch.from_numpy(np.load(self.output_path/self.inventory[index]))

        return clean, noisy, output


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    import utils.config as Config
    from utils.gram_transform import gram_log_modulus_normalize_reverse, ReverseRISpectrograms

    try:
        import simpleaudio as sa
        hasAudio = True
    except ModuleNotFoundError:
        hasAudio = False


    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-debug', '-d', action='store_true')

    args = parser.parse_args()

    dataset_type = 'train'
    # parse argument
    opt = Config.parse(args)

    sample_rate = opt['sample_rate']
    datatype = opt['datatype']

    dataset_opt = opt['datasets'][dataset_type]
    dataroot = '../' + dataset_opt['dataroot']

    encoder_opt = opt['model']['encoder']
    encoder_type = opt['model']['encoder']['type']
    N = encoder_opt[encoder_type]['N']
    L = encoder_opt[encoder_type]['L']
    stride =encoder_opt[encoder_type]['stride']
    expand_order = encoder_opt[encoder_type]['expand_order']

    T = (N-1) * stride

    dataset = AudioDataset(dataroot, datatype)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    clean, noisy, _ = next(iter(dataloader))
    print(clean.shape) # should be [2, 128, 128]


    def play_audio(data: torch.FloatTensor, datatype, sample_rate, N=None, L=None, stride=None, expand_order=None):
        if hasAudio:
            if datatype == '.wav':
                sound = data.numpy()

            elif datatype == '.spec.npy':
                transform = ReverseRISpectrograms(N=N, L=L, stride=stride, sample_rate=sample_rate,
                                                  expand_order=expand_order, use_mel=False)
                sound = torch.squeeze(transform(data)).numpy()
            elif datatype == '.mel.npy':
                transform = ReverseRISpectrograms(N=N, L=L, stride=stride, sample_rate=sample_rate,
                                                  expand_order=expand_order, use_mel=True)
                sound = torch.squeeze(transform(data)).numpy()

            play_obj = sa.play_buffer(sound, 1, 32 // 8, sample_rate)
            play_obj.wait_done()

    def plotTwoGrams(grams):
        fig, axs = plt.subplots(1, 2)
        for i in [0, 1]:
            axs[i].set_ylabel('frequency')
            axs[i].set_xlabel('frame')
            im = axs[i].imshow(grams[i,:,:], origin='lower', aspect='auto')
            fig.colorbar(im, ax=axs[i])

        axs[0].set_title('Real Gram, log, normalized')
        axs[1].set_title('Phase Gram, log, normalized')
        plt.show(block=False)



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


    if datatype == '.mel.npy':


        play_audio(torch.squeeze(clean[0, :, :]), datatype, sample_rate, N, L, stride, expand_order)
        play_audio(torch.squeeze(noisy[0, :, :]), datatype, sample_rate, N, L, stride, expand_order)

        plotTwoGrams(clean[0, :, :, :])
        plotTwoGrams(noisy[0, :, :, :])
        plotSpectrogram(clean[0, :, :, :], expand_order)
        plotSpectrogram(noisy[0, :, :, :], expand_order)
        plt.show()


    elif datatype == '.spec.npy':

        play_audio(torch.squeeze(clean[0, :, :]), datatype, sample_rate, N, L, stride, expand_order)
        play_audio(torch.squeeze(noisy[0, :, :]), datatype, sample_rate, N, L, stride, expand_order)
        plotTwoGrams(clean[0, :, :, :])
        plotTwoGrams(noisy[0, :, :, :])
        plotSpectrogram(clean[0, :, :, :], expand_order)
        plotSpectrogram(noisy[0, :, :, :], expand_order)
        plt.show()
