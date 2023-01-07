import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from .face_image_model import FaceImageModel
from .face_grid_model import FaceGridModel
from .eyes_image_model import EyesImageModel
from .decoder_model import DecoderModel

"""
Pytorch model for the iTracker.

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
"""


class GazeModel(nn.Module):

    def __init__(self):
        super(GazeModel, self).__init__()
        self.face_model = FaceImageModel()
        self.eyes_model = EyesImageModel()
        self.grid_model = FaceGridModel()
        self.decoder_model = DecoderModel()

    def forward(self, faces, left_eyes, right_eyes, face_grids):
        faces = self.face_model(faces)
        eyes = self.eyes_model(left_eyes, right_eyes)
        grids = self.grid_model(face_grids)
        x = torch.cat((faces, eyes, grids), 1)
        x = self.decoder_model(x)
        return x
