import torch.utils.data as data
import scipy.io as sio
from pathlib import Path
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
import re

from . import files as fs
from . import metadata as md

'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.

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

MEAN_PATH = './src/data'


class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, mean_image):
        self.mean_image = transforms.ToTensor()(mean_image / 255)

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor.sub(self.mean_image)


class DataLoader(data.Dataset):

    def __init__(self, data_path: Path, split=md.TRAIN, image_size=(224, 224), grid_size=(25, 25)):

        self.data_path = data_path
        self.image_size = image_size
        self.grid_size = grid_size

        print('Loading gaze tracker dataset...')
        metadata_path = data_path.joinpath(fs.METADATA_CSV)

        #metaFile = 'metadata.mat'
        if metadata_path is None or not metadata_path.is_file():
            raise RuntimeError(
                'There is no such file %s! Provide a valid dataset path.' %
                metadata_path)

        self.metadata: pd.DataFrame = pd.read_csv(metadata_path)
        if self.metadata is None:
            raise RuntimeError(
                'Could not read metadata file %s! Provide a valid dataset path.'
                % metadata_path)

        self.face_mean = self.load_metadata(
            os.path.join(fs.MEAN_PATH, fs.MEAN_FACE_224_MAT))['image_mean']
        self.left_eye_mean = self.load_metadata(
            os.path.join(fs.MEAN_PATH, fs.MEAN_LEFT_224_MAT))['image_mean']
        self.right_eye_mean = self.load_metadata(
            os.path.join(fs.MEAN_PATH, fs.MEAN_RIGHT_224_MAT))['image_mean']

        self.transform_face = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            SubtractMean(mean_image=self.face_mean),
        ])
        self.transform_left_eye = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            SubtractMean(mean_image=self.left_eye_mean),
        ])
        self.transform_right_eye = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            SubtractMean(mean_image=self.right_eye_mean),
        ])

        if split == md.TRAIN:
            self.metadata = self.metadata.loc[self.metadata[md.SPLIT] == split].reindex()
        elif split == md.VALIDATE:
            self.metadata = self.metadata.loc[self.metadata[md.SPLIT] == split].reindex()
        else:
            raise RuntimeError(f'Invalid split value: {split}')

        print(f'Loaded iTracker dataset split {split} with {self.metadata.shape[0]} records...')

    def load_metadata(self, filename, silent=False):
        try:
            # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
            if not silent:
                print('\tReading metadata from %s...' % filename)
            metadata = sio.loadmat(filename,
                                   squeeze_me=True,
                                   struct_as_record=False)
        except:
            print('\tFailed to read the meta file "%s"!' % filename)
            return None
        return metadata

    def load_image(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im

    def make_grid(self, params):
        grid_len = self.grid_size[0] * self.grid_size[1]
        grid = np.zeros([grid_len,], np.float32)
        
        inds_y = np.array([i // self.grid_size[0] for i in range(grid_len)])
        inds_x = np.array([i % self.grid_size[0] for i in range(grid_len)])
        cond_x = np.logical_and(inds_x >= params[0], inds_x < params[0] + params[2]) 
        cond_y = np.logical_and(inds_y >= params[1], inds_y < params[1] + params[3]) 
        cond = np.logical_and(cond_x, cond_y)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):

        recording_index = self.metadata.loc[index, md.RECORDING_INDEX]
        frame_index = self.metadata.loc[index, md.FRAME_INDEX]

        face_path = self.data_path.joinpath(f'{recording_index:05d}',fs.FACE, f'{frame_index:05d}')
        left_eye_path = self.data_path.joinpath(f'{recording_index:05d}',fs.LEFT_EYE, f'{frame_index:05d}')
        right_eye_path = self.data_path.joinpath(f'{recording_index:05d}',fs.RIGHT_EYE, f'{frame_index:05d}')

        face = self.load_image(face_path)
        left_eye = self.load_image(left_eye_path)
        right_eye = self.load_image(right_eye_path)

        face = self.transform_face(face)
        left_eye = self.transform_left_eye(left_eye)
        right_eye = self.transform_right_eye(right_eye)

        face_grid = self.make_grid(self.metadata.loc[index, md.FACE_GRID])

        gaze = np.array([
            self.metadata[index, md.GAZE_X],
            self.metadata[index, md.GAZE_Y],
        ], np.float32)

        # to tensor
        row = torch.LongTensor([int(index)])
        face_grid = torch.FloatTensor(face_grid)
        gaze = torch.FloatTensor(gaze)

        return row, face, left_eye, right_eye, face_grid, gaze

    def __len__(self):
        return self.metadata.shape[0]
