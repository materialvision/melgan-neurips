import torch
import torch.utils.data
import torch.nn.functional as F

import librosa
from librosa.core import load
from librosa.util import normalize

from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class AudioFile:
    path: str
    duration: float

    @classmethod
    def from_path(cls, path):
        duration = librosa.get_duration(filename=path)
        return cls(path, duration)


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, sampling_rate, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        files = files_to_list(training_files)
        files = [Path(training_files).parent / x for x in files]
        self.audio_files = [AudioFile.from_path(path) for path in files]
        self.durations = np.array([a.duration for a in self.audio_files])
        self.duration = self.durations.sum()
        self.weights = self.durations / self.duration
        self.augment = augment
        self.random_state = np.random.RandomState(1234)

    def __getitem__(self, index):

        audio_file = self.random_state.choice(self.audio_files, p=self.weights)
        offset = self.random_state.randint(audio_file.duration)
        audio = self.load_wav_to_torch(audio_file.path, offset)

        # Take segment
        if audio.size(0) < self.segment_length:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0
        return audio.unsqueeze(0)

    def __len__(self):
        return int(self.durations.sum() * self.sampling_rate / self.segment_length)

    def load_wav_to_torch(self, full_path, offset):
        """
        Loads wavdata into torch array
        """
        load_duration = self.segment_length / self.sampling_rate
        data, _ = load(
            full_path, sr=self.sampling_rate, offset=offset, duration=load_duration
        )
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = self.random_state.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).float()
