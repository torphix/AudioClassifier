import json
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from librosa.feature import melspectrogram


class AudioClassificationDataset(Dataset):
    def __init__(self, config, n_classes):
        super().__init__()
        '''
        Expects path to wav folder
        & Text doc with wav ID + speaker ID
        file_id|text|speaker_id
        '''
        self.wav_dir = config['wav_dir']
        self.sample_rate = config['sample_rate']
        self.max_wav_len = config['max_wav_len']
        # Mel features
        self.n_fft = config['n_fft']
        self.hop_length = config['hop_length']
        self.win_length = config['win_length']
        self.n_classes = n_classes

        # Get speaker IDS
        with open(config['metadata_path'], 'r') as f:
            lines = f.readlines()
        self.speaker_ids = [line.split("|")[-1].strip('\n') for line in lines]
        self.file_ids = [line.split("|")[0] for line in lines]

        print(f'Number of speakers found: {len(set(self.speaker_ids))}')
        # Assign each speaker to a key
        self.speaker_keys = {speaker: i for i,
                             speaker in enumerate(set(self.speaker_ids))}
        assert n_classes == len(self.speaker_keys.values()), \
            f"Differing desired number of classes {n_classes} & actual {len(self.speaker_keys)}"
        with open(config['speaker_id_save_path'], 'w') as f:
            f.write(json.dumps(self.speaker_keys))

    @property
    def audio_labels(self):
        return self.speaker_keys

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        wav_path = f'{self.wav_dir}/{self.file_ids[idx]}.wav'
        wav, sr = librosa.load(wav_path,
                               sr=self.sample_rate)
        wav, wav_idx = librosa.effects.trim(wav)
        max_len = int(self.sample_rate * self.max_wav_len)
        wav = torch.tensor(wav[:max_len])
        if wav.shape[0] < max_len:
            wav = F.pad(wav, (0, max_len-wav.shape[0]), value=0)
        mel = melspectrogram(np.array(wav),
                             sr=self.sample_rate,
                             n_fft=self.n_fft,
                             hop_length=self.hop_length,
                             win_length=self.win_length)

        speaker_id = torch.zeros((self.n_classes))
        speaker_key = self.speaker_keys[self.speaker_ids[idx]]
        speaker_id[speaker_key] = 1
        mel = torch.tensor(mel)
        mel = mel / torch.max(mel)
        return mel, speaker_id, self.file_ids[idx]