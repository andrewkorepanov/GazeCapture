import argparse
from pathlib import Path

from . import config
from .models import GazeModel
from .training import Train

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gaze Model Learning')

    parser.add_argument('-s',
                        action='store_true',
                        default=False,
                        help="Just sink and terminate.")
    parser.add_argument('-r',
                        action='store_true',
                        default=False,
                        help="Start from scratch (do not load checkpoints).")

    parser.add_argument(
        "data_path",
        metavar="data_path",
        type=Path,
        help=
        "Path to extracted dataset files. It should have folders called '%%05d' in it."
    )
    args = parser.parse_args()

    data_path = args.data_path
    reset = args.r
    sink = args.s

    print('DATA_PATH: ', data_path)
    print('RESET: ', reset)
    print('SINK: ', sink)

    model = GazeModel()
    train = Train(data_path, config.EPOCHS, config.LEARNING_RATE,
                  config.MOMENTUM, config.WEIGHT_DECAY, config.WORKERS, reset,
                  sink)
    train(model)

    print('DONE')
