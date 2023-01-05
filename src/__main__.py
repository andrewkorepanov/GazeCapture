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

from . import config
from . import Training
from . import GazeModel, DecoderModel


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

    parser = argparse.ArgumentParser(description='Gaze Model Learning')
    parser.add_argument('--data_path', default='./src', type=Path, help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.")
    parser.add_argument('--sink', action='store_true', default=False, help="Just sink and terminate.")
    parser.add_argument('--reset', action='store_true', default=False, help="Start from scratch (do not load checkpoints).")
    args = parser.parse_args()

    data_path = args.data_path
    reset = args.reset
    sink = args.sink

    training = Training(data_path, config.EPOCHS, config.LEARNING_RATE, config.WORKERS, reset, sink)
    model = GazeModel()
    
    training(model)

    print('DONE')
