import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from .image_model import ImageModel

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


class EyesImageModel(nn.Module):
    
    def __init__(self):
        super(EyesImageModel, self).__init__()
        self.conv = ImageModel()
        self.fc = nn.Sequential(
            nn.Linear(2*12*12*64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, left_eye, right_eye):
        left_eye = self.conv(left_eye)
        right_eye = self.conv(right_eye)
        eyes = torch.cat((left_eye, right_eye), 1)
        eyes = self.fc(eyes)
        return eyes
