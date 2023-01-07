import argparse
from pathlib import Path

from .prepare_dataset import PrepareDataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gaze Capture Data Prepare')
    parser.add_argument(
        '-o',
        type=Path,
        default=None,
        help=
        "Where to write the output. Can be the same as dataset_path if you wish (=default)."
    )
    parser.add_argument(
        "dataset_path",
        metavar="dataset_path",
        type=Path,
        help=
        "Path to extracted dataset files. It should have folders called '%%05d' in it."
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.o

    if output_path is None:
        output_path = dataset_path

    if not dataset_path or not dataset_path.is_dir():
        raise RuntimeError('No such dataset folder %s!' % args.dataset_path)

    prepare = PrepareDataset(dataset_path, output_path)
    prepare()

    print('DONE')
