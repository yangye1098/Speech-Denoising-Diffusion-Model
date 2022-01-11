import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from random import shuffle
import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

try:
    import simpleaudio as sa
    hasAudio = True
except ModuleNotFoundError:
    hasAudio = False

def generate_inventory(path, file_type='.wav'):
    path = Path(path)
    assert path.is_dir(), '{:s} is not a valid directory'.format(path)

    file_paths = path.glob('*'+file_type)
    file_names = [ file_path.name for file_path in file_paths ]
    assert file_names, '{:s} has no valid {} file'.format(path, file_type)
    shuffle(file_names)
    return file_names

class AudioDataset(Dataset):
    def __init__(self, dataroot, datatype, snr, sample_rate=8000, T=-1):
        if datatype not in ['.wav', '.spec.npy', '.mel.npy']:
            raise NotImplementedError
        self.datatype = datatype
        self.snr = snr
        self.sample_rate = sample_rate
        self.T = T

        self.clean_path = Path('{}/clean'.format(dataroot))
        self.noisy_path = Path('{}/noisy_{}'.format(dataroot, snr))

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

    def to_audio(self, grams, N):
        """
        similar to STFTDecoder.decode
        :param grams: [C, N, N]
        :return:
        """
        assert N == grams.shape[1] and N == grams.shape[2]

        Zxx = torch.zeros([1, N+1, N], dtype=torch.cfloat)
        z_temp = torch.movedim(grams, 0, 2)
        z_temp = 10 ** (z_temp * 10 - 10)

        Zxx[:, 1:, : ] = torch.view_as_complex(z_temp.contiguous())
        _, sound = signal.istft(Zxx, self.sample_rate)
        return sound

    def getName(self, idx):
        return self.inventory[idx]


    def playIdx(self, idx, N):
        if hasAudio:
            clean, noisy = self.__getitem__(idx)
            if self.datatype == '.wav':
                clean_sound = clean.numpy()
                noisy_sound = noisy.numpy()

            elif self.datatype == '.spec.npy':
                clean_sound = self.to_audio(clean, N)
                noisy_sound = self.to_audio(noisy, N)

            play_obj = sa.play_buffer(clean_sound, 1, 32//8, self.sample_rate)
            play_obj.wait_done()
            play_obj = sa.play_buffer(noisy_sound, 1, 32//8, self.sample_rate)
            play_obj.wait_done()


class OutputDataset(AudioDataset):
    def __init__(self, dataroot, datatype, snr, sample_rate=8000, T=-1):
        super().__init__(dataroot, datatype, snr, sample_rate, T)

        self.noisy_path = Path('{}/noisy'.format(dataroot))
        self.output_path = Path('{}/output'.format(dataroot))


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

    def to_audio(self, grams, N):
        """
        similar to STFTDecoder.decode
        :param grams: [C, N, N]
        :return:
        """
        assert N == grams.shape[1] and N == grams.shape[2]

        Zxx = torch.zeros([1, N+1, N], dtype=torch.cfloat)
        z_temp = torch.movedim(grams, 0, 2)
        z_temp = 10 ** (z_temp * 10 - 10)

        Zxx[:, 1:, : ] = torch.view_as_complex(z_temp.contiguous())
        _, sound = signal.istft(Zxx, self.sample_rate)
        return sound

    def saveIdxToWav(self, idx, N):
        clean, noisy, output = self.__getitem__(idx)
        if self.datatype == '.spec.npy':
            clean_sound = self.to_audio(clean, N)
            wavfile.write(f'{self.clean_path / self.inventory[idx]}.wav', self.sample_rate, clean_sound.T)
            noisy_sound = self.to_audio(noisy, N)
            wavfile.write(f'{self.noisy_path / self.inventory[idx]}.wav', self.sample_rate, noisy_sound.T)
            output_sound = self.to_audio(output, N)
            wavfile.write(f'{self.output_path / self.inventory[idx]}.wav', self.sample_rate, output_sound.T)

    def playIdx(self, idx, N):
        if hasAudio:
            clean, noisy, output = self.__getitem__(idx)
            if self.datatype == '.wav':
                clean_sound = clean.numpy()
                noisy_sound = noisy.numpy()
                output_sound = output.numpy()

            elif self.datatype == '.spec.npy':
                clean_sound = self.to_audio(clean, N)
                noisy_sound = self.to_audio(noisy, N)
                output_sound = self.to_audio(output, N)


            play_obj = sa.play_buffer(clean_sound, 1, 32//8, self.sample_rate)
            play_obj.wait_done()
            play_obj = sa.play_buffer(noisy_sound, 1, 32//8, self.sample_rate)
            play_obj.wait_done()
            play_obj = sa.play_buffer(output_sound, 1, 32//8, self.sample_rate)
            play_obj.wait_done()






if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import librosa

    N = 128
    #L = 256
    #stride = 128
    sample_rate = 8000
    snr = 0
    dataroot = f'data/wsj0_si_tr_{snr}'
    datatype = '.spec.npy'
    dataset_tr = AudioDataset(dataroot, datatype, snr)
    #dataset_tr.playIdx(0, N)

    dataloader = DataLoader(dataset_tr, batch_size=2)
    clean, noisy = next(iter(dataloader))
    print(clean.shape) # should be [2, 128, 128]

    def plotMel(gram):
        fig, axs = plt.subplots(1, 1)
        axs.set_title('MelSpectrogram log, normalized ')
        axs.set_ylabel('Mel Frequency')
        axs.set_xlabel('frame')
        im = axs.imshow(gram, origin='lower', aspect='auto')
        fig.colorbar(im, ax=axs)
        plt.show(block=False)


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


    def plotSpectrogram(grams):
        grams_original = 10**(10 * grams - 10)
        spectrogram = np.sum([np.square(grams_original[0, :, :]), np.square(grams_original[1, :, :])])

        plt.figure()
        im = plt.imshow(20*np.log10(spectrogram), origin='lower', aspect='auto')
        plt.colorbar(im)
        plt.xlabel('Frame')
        plt.ylabel('Frequency')
        plt.title('Spectrogram, db re 1')
        plt.show(block=False)

    if datatype == '.mel.npy':
        plotMel(clean[0, 0, :, :])
        plotMel(noisy[0, 0, :, :])
        plt.show()


    elif datatype == '.spec.npy':

        plotTwoGrams(clean[0, :, :, :])
        plotTwoGrams(noisy[0, :, :, :])
        #clean_sound = dataset_tr.to_audio(clean[0, :, :, :], N)
        #plt.figure()
        #plt.specgram(np.squeeze(clean_sound), Fs=sample_rate, NFFT=N+1)
        #plt.show()
        plotSpectrogram(clean[0, :, :, :])
        plotSpectrogram(noisy[0, :, :, :])
        plt.show()
