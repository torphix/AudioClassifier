import os
import yaml
import json
import torch
import librosa
import numpy as np
import torch.nn as nn
from datetime import datetime
import pytorch_lightning as ptl
import torch.nn.functional as F
from librosa.feature import melspectrogram
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split


class SpeechClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(config['k_sizes'])-1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(config['hid_sizes'][i],
                              config['hid_sizes'][i+1],
                              config['k_sizes'][i]),
                    nn.ReLU(),
                    nn.BatchNorm1d(config['hid_sizes'][i+1])
                ))
        self.out_layer = nn.Linear(config['hid_sizes'][-1]*116,
                                   config['n_classes'])
        self.layers = nn.Sequential(*self.layers)

    def forward(self, wavs):
        x = self.layers(wavs)
        BS, N, L = x.size()
        x = x.view(BS, -1)
        x = self.out_layer(x)
        return x


class SpeakerClassificationDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        '''
        Expects path to wav folder
        & Text doc with wav ID + speaker ID
        file_id|text|speaker_id
        '''
        self.wav_dir = config['wav_dir']
        self.sample_rate = config['sample_rate']
        self.max_wav_len = config['max_wav_len']
        self.n_classes = config['n_classes']

        # Get speaker IDS
        with open(config['metadata_path'], 'r') as f:
            lines = f.readlines()
        self.speaker_ids = [line.split("|")[-1].strip('\n') for line in lines]
        self.file_ids = [line.split("|")[0] for line in lines]

        print(f'Number of speakers found: {len(set(self.speaker_ids))}')
        # Assign each speaker to a key
        self.speaker_keys = {speaker: i for i,
                             speaker in enumerate(set(self.speaker_ids))}
        assert config['n_classes'] == len(self.speaker_keys.values()), \
            f"Differing desired number of classes {config['n_classes']} & actual {len(self.speaker_keys)}"
        with open(config['speaker_id_path'], 'w') as f:
            f.write(json.dumps(self.speaker_keys))

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        wav_path = f'{self.wav_dir}/{self.file_ids[idx]}.wav'
        wav, sr = librosa.load(wav_path,
                               sr=self.sample_rate)
        wav = torch.tensor(wav[:self.max_wav_len])
        if wav.shape[0] < self.max_wav_len:
            wav = F.pad(wav, (0, self.max_wav_len-wav.shape[0]), value=0)
        mel = melspectrogram(np.array(wav),
                             sr=self.sample_rate,
                             n_fft=2048,
                             hop_length=512,
                             win_length=50)

        speaker_id = torch.zeros((self.n_classes))
        speaker_key = self.speaker_keys[self.speaker_ids[idx]]
        speaker_id[speaker_key] = 1
        mel = torch.tensor(mel)
        mel = mel / torch.max(mel)
        return mel, speaker_id


class SpeechClassifierModule(ptl.LightningModule):
    def __init__(self, module_config, model_config, data_config):
        super().__init__()
        with open(module_config, 'r') as f:
            self.module_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(model_config, 'r') as f:
            model_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(data_config, 'r') as f:
            self.data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.module_config = module_config
        self.data_config = data_config

        self.model = SpeechClassifier(model_config)
        self.dataset = SpeakerClassificationDataset(self.data_config)
        self.train_ds, self.val_ds = random_split(self.dataset,
                                                  self.data_config['dataset_split'])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            **self.data_config['dataloader'])

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            **self.data_config['dataloader'])

    def forward(self, batch):
        outputs = self.model(batch)
        return outputs

    def training_step(self, batch, batch_idx):
        wavs, targets = batch
        outputs = self.forward(wavs)
        outputs = F.softmax(outputs, dim=1)
        loss = F.cross_entropy(outputs, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        wavs, targets = batch
        outputs = self.forward(wavs)
        outputs = F.softmax(outputs, dim=1)
        loss = F.cross_entropy(outputs, targets)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.module_config['learning_rate'])
        return {'optimizer': optimizer,
                'scheduler': ReduceLROnPlateau(optimizer)}


def train(module_config,
          model_config,
          data_config,
          trainer_config,
          pretrained_path=None):

    module = SpeechClassifierModule(
        module_config, model_config, data_config)
    if pretrained_path is not None:
        module.model.load_state_dict(torch.load(pretrained_path))
    with open(trainer_config, 'r') as f:
        trainer_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    trainer = ptl.Trainer(**trainer_config)
    trainer.fit(module)
    os.makedirs('pretrained_models', exist_ok=True)
    timestamp = datetime.now().strftime("%d/%m/%y")
    save_name = f'e{trainer_config["max_epochs"]}_classifer_{timestamp}.pth'
    torch.save(module.model.state_dict(),
               f'pretrained_models/{save_name}')


def inference(
        model_config,
        data_config,
        n_classes,
        pretrained_path,
        wav_path):
    with open(model_config, 'r') as f:
        model_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(data_config, 'r') as f:
        data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model = SpeechClassifier(model_config, n_classes)
    model.load_state_dict(torch.load(pretrained_path))

    wav, sr = librosa.load(wav_path,
                           sr=data_config['sample_rate'])
    wav = torch.tensor(wav[:data_config['max_wav_len']])
    wav = F.pad(wav, (0, data_config['max_wav_len']-wav.shape[0]), value=0)
    mel = melspectrogram(np.array(wav),
                         sr=data_config['sample_rate'],
                         n_fft=data_config['n_fft'],
                         hop_length=data_config['hop_length'],
                         win_length=data_config['win_length'])

    model.forward
