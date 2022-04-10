# Modular Audio Classifier
- Designed to be customisable for any number of audio classes
- Was created as part of a data processing pipline to automate labelling of audio files for part of a wider project

## How it works
- Series of 1 dimensional convolutional layers to extract features
- Classifies the extracted latent variable as belonging to a particular class using cross entropy loss

## Setup
- Create new python / conda env
- pip install -r requirments.txt

## Custom training
1. Open config folder
- Model config allows you to edit layers / model hyper parameters
- Data config allows you to edit dataloader & data loader paramteres
- Module config allows you to edit number of target classes, trainer, optimization parameters

2. Data formatting
- Provide wav files in folder
- Provide metadata in text file format: ${WavFileName}|${text}|${labelName}, eg: 1_wavfile|How are you today|happy
- Ensure to correctly specify the wav folder in data_config.yaml

3. Call python main.py train
- Classes configs are automatically read so ensure to adjust before training to your desired paramters

## Limitations & Future work
- Currently only supports 5 second audio clips sampled at 16000
- In the future will support variable length audio clips 




