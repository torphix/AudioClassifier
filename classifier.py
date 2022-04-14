import os
import yaml
import torch
import librosa
import numpy as np
import torch.nn as nn
from datetime import datetime
import pytorch_lightning as ptl
import torch.nn.functional as F
from utils import filter_incorrect
from librosa.feature import melspectrogram
from dataset import AudioClassificationDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split


class AudioClassifier(nn.Module):
    def __init__(self, config, n_classes):
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
        self.out_layer = nn.Linear(6784,
                                   n_classes)
        self.layers = nn.Sequential(*self.layers)

    def _calc_output_size(self, w, k, p, s):
        return ((w-k+(2*p))/s) + 1

    def forward(self, wavs):
        x = self.layers(wavs)
        BS, N, L = x.size()
        x = x.view(BS, -1)
        x = self.out_layer(x)
        return x
    

class AudioClassifierModule(ptl.LightningModule):
    def __init__(self, module_config, model_config, data_config):
        super().__init__()
        with open(module_config, 'r') as f:
            self.module_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(model_config, 'r') as f:
            model_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(data_config, 'r') as f:
            self.data_config = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.model = AudioClassifier(model_config, self.module_config['n_classes'])
        self.dataset = AudioClassificationDataset(self.data_config, 
                                                    self.module_config['n_classes'])
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
        wavs, targets, wav_ids = batch
        outputs = self.forward(wavs)
        outputs = F.softmax(outputs, dim=1)
        loss = F.cross_entropy(outputs, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        wavs, targets, wav_ids = batch
        outputs = self.forward(wavs)
        outputs = F.softmax(outputs, dim=1)
        loss = F.cross_entropy(outputs, targets)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=float(self.module_config['learning_rate']))
        return {'optimizer': optimizer,
                'scheduler': ReduceLROnPlateau(optimizer)}


def train(module_config,
          model_config,
          data_config,
          trainer_config,
          pretrained_path=None):

    module = AudioClassifierModule(
        module_config, model_config, data_config)
    if pretrained_path is not None:
        print('Loading from checkpoint')
        module.model.load_state_dict(torch.load(pretrained_path))
    with open(trainer_config, 'r') as f:
        trainer_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    trainer = ptl.Trainer(**trainer_config)
    trainer.fit(module)
    os.makedirs('pretrained_models', exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%y")
    save_name = f'e{trainer_config["max_epochs"]}_classifer_{timestamp}.pth'
    torch.save(module.model.state_dict(),
               f'pretrained_models/{save_name}')
    

def inference(
        model_config,
        data_config,
        module_config,
        pretrained_path,
        wav_path):
    with open(model_config, 'r') as f:
        model_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(data_config, 'r') as f:
        data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(module_config, 'r') as f:
        module_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model = AudioClassifierModule(model_config, module_config['n_classes'])
    model.load_state_dict(torch.load(pretrained_path))

    wav, sr = librosa.load(wav_path,
                           sr=data_config['sample_rate'])
    max_len = int(data_config['max_wav_len'] * data_config['sample_rate'])
    wav = torch.tensor(wav[:max_len])
    wav = F.pad(wav, (0, max_len-wav.shape[0]), value=0)
    mel = melspectrogram(np.array(wav),
                         sr=data_config['sample_rate'],
                         n_fft=data_config['n_fft'],
                         hop_length=data_config['hop_length'],
                         win_length=data_config['win_length'])
    output = model.forward(torch.tensor(mel).unsqueeze(0))
    output = F.softmax(output)
    return output


def validate(
        model_config,
        data_config,
        module_config,
        pretrained_path):
    '''
    Iterates over entire dataset specified in data_config, returning
    incorrect classifications & confusion matrix
    '''
    with open(model_config, 'r') as f:
        model_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(data_config, 'r') as f:
        data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(module_config, 'r') as f:
        module_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    dataset = AudioClassificationDataset(data_config, module_config['n_classes'])
    dataloader = DataLoader(dataset, **data_config['dataloader'])
    model = AudioClassifier(model_config, module_config['n_classes'])
    model.load_state_dict(torch.load(pretrained_path))

    for batch in dataloader:
        mels, ids, wav_ids = batch[0], batch[1], batch[2]
        outputs = model.forward(mels)
        outputs = F.softmax(outputs)
        incorrect = filter_incorrect(outputs, ids, wav_ids)
        print(incorrect)
    return incorrect


