from io import BytesIO
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
from . import SND_DTYPE
from .util import generate_inventory, load_wav
from pathlib import Path


try:
    import simpleaudio as sa
    hasAudio = True
except ModuleNotFoundError:
    hasAudio = False


class AudioDataset(Dataset):
    def __init__(self, dataroot, datatype, snr, T, sample_rate=8000, split='train', data_len=-1):
        self.datatype = datatype
        self.data_len = data_len
        self.snr = snr
        self.T = T
        self.sample_rate = sample_rate
        self.split = split


        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'wav':
            self.clean_path = Path('{}/clean'.format(dataroot))
            self.noisy_path = Path('{}/noisy_{}'.format(dataroot, snr))
            self.inventory = generate_inventory( self.clean_path, datatype)
            self.dataset_len = len(self.inventory)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                clean_snd_bytes = txn.get(
                    'clean_{:d}'.format(index).encode('utf-8')
                )
                noisy_snd_bytes = txn.get(
                    'noisy_{}_{:d}'.format(
                        self.snr, index).encode('utf-8')
                )

                clean_snd = np.frombuffer(clean_snd_bytes, dtype=SND_DTYPE)
                noisy_snd = np.frombuffer(noisy_snd_bytes, dtype=SND_DTYPE)
        else:
            sr, clean_snd = load_wav(self.clean_path/self.inventory[index], self.T)
            assert(sr==self.sample_rate)
            sr, noisy_snd = load_wav(self.noisy_path/self.inventory[index], self.T)
            assert (sr == self.sample_rate)
        return {'Clean': clean_snd, 'Noisy': noisy_snd, 'Index': index}

    def playIdx(self, idx):
        if hasAudio:
            item = self.__getitem__(idx)
            play_obj = sa.play_buffer(item['Clean'].numpy(), 1, 32//8, self.sample_rate)
            play_obj.wait_done()
            play_obj = sa.play_buffer(item['Noisy'].numpy(), 1, 32//8, self.sample_rate)
            play_obj.wait_done()



if __name__ == '__main__':
    snr = 5
    dataroot = f'./data/wsj0_si_tr_{snr}'
    datatype = 'wav'
    dataset_tr = AudioDataset(dataroot, datatype, snr)
    dataset_tr.playIdx(0)
