import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from .prepare_dataset import PrepareDataset
'''
Train/test code for iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
'''

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
