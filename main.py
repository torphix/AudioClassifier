import os
import sys
import argparse
from classifier import inference, train, validate


if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'train':
        root_path = os.path.abspath('.')
        train(f'{root_path}/configs/module.yaml',
              f'{root_path}/configs/model.yaml',
              f'{root_path}/configs/data.yaml',
              f'{root_path}/configs/trainer.yaml')

    elif command == 'finetune':
        parser.add_argument('-pth', '--model_path', required=True,
                            help='Path to pretrained model')
        args, leftover_args = parser.parse_known_args()
        root_path = os.path.abspath('.')
        train(f'{root_path}/configs/module.yaml',
              f'{root_path}/configs/model.yaml',
              f'{root_path}/configs/data.yaml',
              f'{root_path}/configs/trainer.yaml',
              pretrained_path=args.model_path)
        
    elif command == 'inference':
        parser.add_argument('-pth', '--model_path', required=True,
                            help='Path to pretrained model')
        parser.add_argument('-wp', '--wav_path', required=True)
        parser.add_argument('-sid', '--speaker_ids_path', required=False)
        args, leftover_args = parser.parse_known_args()
        root_path = os.path.abspath('.')
        output = inference(
                f'{root_path}/configs/model.yaml',
                f'{root_path}/configs/data.yaml',
                f'{root_path}/configs/module.yaml',
                args.model_path,
                args.wav_path)

    elif command == 'validate':
        parser.add_argument('-pth', '--model_path', required=True,
                            help='Path to pretrained model')
        args, leftover_args = parser.parse_known_args()
        root_path = os.path.abspath('.')
        output = validate(
                f'{root_path}/configs/model.yaml',
                f'{root_path}/configs/data.yaml',
                f'{root_path}/configs/module.yaml',
                args.model_path)

# TODO make mel spec hparams customizable and max len customaizable
