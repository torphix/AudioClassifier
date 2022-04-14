import os
import sys
import argparse
from classifier import inference, train


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
        parser.add_argument('-pth', '--pretrained_path', required=True,
                            help='Path to pretrained model')
        args, leftover_args = parser.parse_known_args()
        inference(
              './configs/module.yaml',
              './configs/model.yaml',
              './configs/data.yaml',
              './configs/trainer.yaml',
              args.pretrained_path)

    elif command == 'inference':
        pass
# TODO make mel spec hparams customizable and max len customaizable
