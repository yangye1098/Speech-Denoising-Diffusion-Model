import argparse
from io import BytesIO
import multiprocessing
from functools import partial
from tqdm import tqdm
import os
from pathlib import Path
import lmdb
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

from data import SND_DTYPE

def rmsNormalize(sound):
    sound = sound.astype(SND_DTYPE)
    # rms equalize
    rms = np.sqrt(np.mean(sound**2))
    sound = sound/rms
    return sound

def cropOrPad(sound, sampleRate, sndDuration):
    targetLength = int(sampleRate*sndDuration)
    if sound.shape[0] >= targetLength:
        return sound[:targetLength]
    else:
        zeroLength = targetLength - sound.shape[0]
        return np.pad(sound, (0, zeroLength), 'constant')



def addnoiseSNR (sound, noise, snr):
    """
    sound and noise are rms normalized, with same length
    """
    weight = 10 ** (snr/ 20)
    return weight*sound + noise


def addnoise_worker(wav_file, noise, snd_duration,  snr, sample_rate, min_duration):
    """
    open original 16bits wav_file, ignore sound that are too short
    sample it to "sampleRate",
    noise has to be rms normalized
    rms normalize,
    restore back to 16 bits wav file.
    """
    # open
    sr, sound = wavfile.read(wav_file)

    # check length
    snd_length = sound.shape[0]
    if snd_length < min_duration * sr:
        return None, [None, None]
    # resample
    resample_length = int(snd_length / sr * sample_rate)
    sound = resample(sound, resample_length)
    # make the sound same length
    sound = cropOrPad(sound, sample_rate, snd_duration)

    # rms normalize
    sound = rmsNormalize(sound)
    #
    if noise is None:
        noise = np.random.randn(int(sample_rate * snd_duration))
        noise = rmsNormalize(noise)

    mixture = addnoiseSNR(sound, noise, snr)

    # scale to [-1,1] float array

    scale_factor = np.max([np.abs(sound), np.abs(mixture)])
    sound = sound/scale_factor
    mixture = mixture/scale_factor

    return wav_file.name.split('.')[0], [mixture, sound]


def prepare(snd_path, out_path, noise, snd_duration, n_worker=2, snr=-5, sample_rate=8000, min_duration=3, lmdb_save=False):
    addnoise_fn = partial(addnoise_worker, noise=noise, snd_duration=snd_duration,  snr=snr, sample_rate=sample_rate, min_duration=min_duration)

    # all files under one directory allowing one level of subdirectory
    files = [p for p in Path(
        '{}'.format(snd_path)).glob(f'**/*.wav')]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/noisy_{}'.format(out_path, snr), exist_ok=True)
        os.makedirs('{}/clean'.format(out_path), exist_ok=True)
    else:
        # map size 20 G
        env = lmdb.open(out_path, map_size= 20 * (1024 ** 3), readahead=False)

    snd_idx = 0
    if n_worker > 1:
        with multiprocessing.Pool(n_worker) as pool:
            for name, sounds in tqdm(pool.imap_unordered(addnoise_fn, files)):
                noisy_snd, clean_snd = sounds
                if name is not None:
                    if not lmdb_save:
                        # indexing according to name
                        wavfile.write('{}/noisy_{}/{}.wav'.format(out_path, snr, name), sample_rate, noisy_snd)
                        wavfile.write('{}/clean/{}.wav'.format(out_path, name), sample_rate, clean_snd)
                    else:
                        # indexing according to index
                        with env.begin(write=True) as txn:
                            txn.put('noisy_{}_{:d}'.format(
                                snr, snd_idx).encode('utf-8'), noisy_snd)
                            txn.put('clean_{:d}'.format(
                                snd_idx).encode('utf-8'), clean_snd)
                    snd_idx += 1

            if lmdb_save:
                with env.begin(write=True) as txn:
                    txn.put('length'.encode('utf-8'), str(snd_idx).encode('utf-8'))
    else: # debug purpose
        # some may get rejected
        for i, wav_file in enumerate(files):
            name, sounds = addnoise_fn(wav_file)
            noisy_snd, clean_snd = sounds
            if name is not None:
                if not lmdb_save:
                    wavfile.write('{}/noisy_{}/{}.wav'.format(out_path, snr, name), sample_rate, noisy_snd)
                    wavfile.write('{}/clean/{}.wav'.format(out_path, name), sample_rate, clean_snd)
                else:
                    with env.begin(write=True) as txn:
                        txn.put('noisy_{}_{:d}'.format(
                            snr, snd_idx).encode('utf-8'), noisy_snd)
                        txn.put('clean_{:d}'.format(
                            snd_idx).encode('utf-8'), clean_snd)

                snd_idx += 1
        if lmdb_save:
            with env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(snd_idx).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str)
    parser.add_argument('--out', '-o', type=str)
    parser.add_argument('--snr', type=int, default=-5)
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--n_worker', type=int, default=1)
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    # snrs = [int(s.strip()) for s in args.snr.split(',')]

    args.out = '{}_{}'.format(args.out, args.snr)
    # generate noise, change noise type in the future
    noise = None
    prepare(args.path, args.out, noise, snd_duration=args.duration, n_worker=args.n_worker,
            snr=args.snr, sample_rate=args.sample_rate, lmdb_save=args.lmdb)
