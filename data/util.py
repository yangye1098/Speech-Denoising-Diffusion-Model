import os
import torch.utils.data
import numpy as np
from scipy.io import wavfile
from random import shuffle

from . import SND_DTYPE

def load_wav(path, T):
    """
    load wav file to floatTensor from path
    only use first T sample points
    """
    sr, clean_snd = wavfile.read(path)
    clean_snd = torch.FloatTensor(clean_snd[:T].copy())
    return sr, torch.unsqueeze(clean_snd, 0)


def save_wav(wav_tensor:torch.FloatTensor, sr, path):
    """
    save floatTensor to wav file
    """
    # wav_tensor shape [1, 1, T]
    wav_array = torch.squeeze(wav_tensor).numpy()

    if SND_DTYPE == 'float32':
        wavfile.write(path ,sr,np.ascontiguousarray(wav_array, dtype='float32'))
    else:
        raise NotImplementedError

def generate_inventory(path, sound_type):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    snd_names = []
    for dirpath, _, fnames in os.walk(path):
        for fname in fnames:
            _, ext = os.path.splitext(fname)
            if ext.lower() == '.'+sound_type:
                snd_names.append(fname)
    assert snd_names, '{:s} has no valid sound file'.format(path)
    shuffle(snd_names)
    return snd_names

