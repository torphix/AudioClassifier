import sys
import argparse
from classifier import train


if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'train':
        train('./configs/module_config.yaml',
              './configs/model_config.yaml',
              './configs/data_config.yaml',
              './configs/trainer_config.yaml')

    elif command == 'finetune':
        parser.add_argument('-pth', '--pretrained_path', required=True,
                            help='Path to pretrained model')
        args, leftover_args = parser.parse_known_args()
        train('./configs/module_config.yaml',
              './configs/model_config.yaml',
              './configs/data_config.yaml',
              './configs/trainer_config.yaml',
              args.pretrained_path)

    elif command == 'inference':
        pass
# TODO make mel spec hparams customizable and max len customaizable
