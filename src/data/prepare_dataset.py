from typing import Any
import shutil, os, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from . import files as fs
from . import metadata as md

'''
Prepares the GazeCapture dataset for use with the pytorch code. Crops images, compiles JSONs into metadata.mat

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


class PrepareDataset:

    def __init__(self, dataset_path: Path, output_path: Path) -> None:
        self.dataset_path = dataset_path
        self.output_path = output_path

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        self.prepare_path(self.output_path)

        # list recordings
        recordings = np.array(list(self.dataset_path.glob('*')))
        recordings = recordings[[r.is_dir() for r in recordings]]
        recordings.sort()

        # Output structure
        metadata = {
            md.RECORDING_INDEX: [],
            md.FRAME_INDEX: [],
            md.GAZE_X: [],
            md.GAZE_Y: []
        }

        for i, recording in enumerate(recordings):
            print('[%d/%d] Processing recording %s (%.2f%%)' %
                  (i, len(recordings), recording, i / len(recordings) * 100))

            recording_index = int(recording.name)

            recording_dir: Path = recording
            output_dir: Path = self.output_path.joinpath(recording.name)

            # Read info JSONs
            face_info = self.read_json(
                recording_dir.joinpath(fs.APPLE_FACE_JSON))
            if face_info is None:
                continue
            left_eye_info = self.read_json(
                recording_dir.joinpath(fs.APPLE_LEFT_EYE_JSON))
            if left_eye_info is None:
                continue
            right_eye_info = self.read_json(
                recording_dir.joinpath(fs.APPLE_RIGHT_EYE_JSON))
            if right_eye_info is None:
                continue
            dot_info = self.read_json(recording_dir.joinpath(fs.DOT_INFO_JSON))
            if dot_info is None:
                continue
            face_grid_info = self.read_json(recording_dir.joinpath(fs.FACE_GRID_JSON))
            if face_grid_info is None:
                continue
            frames_info = self.read_json(recording_dir.joinpath(fs.FRAMES_JSON))
            if frames_info is None:
                continue

            # Output files
            output_face_path = self.prepare_path(output_dir.joinpath(fs.FACE))
            output_left_eye_path = self.prepare_path(
                output_dir.joinpath(fs.LEFT_EYE))
            output_right_eye_path = self.prepare_path(
                output_dir.joinpath(fs.RIGHT_EYE))

            # Preprocess
            all_valid = np.logical_and(
                np.logical_and(face_info['IsValid'],
                               face_grid_info['IsValid']),
                np.logical_and(left_eye_info['IsValid'],
                               right_eye_info['IsValid']))

            if not np.any(all_valid):
                continue

            frames_info = np.array([
                int(re.match('(\d{5})\.jpg$', x).group(1)) for x in frames_info
            ])

            bbox_from_json = lambda data: np.stack(
                (data['X'], data['Y'], data['W'], data['H']), axis=1).astype(
                    int)

            # for compatibility with matlab code
            face_bbox = bbox_from_json(face_info) + [-1, -1, 1, 1]
            left_eye_bbox = bbox_from_json(left_eye_info) + [0, -1, 0, 0]
            right_eye_bbox = bbox_from_json(right_eye_info) + [0, -1, 0, 0]
            face_grid_bbox = bbox_from_json(face_grid_info)
            # relative to face
            left_eye_bbox[:, :2] += face_bbox[:, :2]
            right_eye_bbox[:, :2] += face_bbox[:, :2]


            for index, frame_index in enumerate(frames_info):
                # Can we use it?
                if not all_valid[index]:
                    continue

                # Load image
                frame_file: Path = recording_dir.joinpath(
                    'frames', f'{frame_index:05d}.jpg')
                if not frame_file.is_file():
                    self.log_error('Warning: Could not read image file %s!' %
                                   frame_file)
                    continue
                frame_image = Image.open(frame_file)
                if frame_image is None:
                    self.log_error('Warning: Could not read image file %s!' %
                                   frame_file)
                    continue
                frame_image = np.array(frame_image.convert('RGB'))

                # Crop images
                face_image = self.crop_image(frame_image, face_bbox[index, :])
                left_eye_image = self.crop_image(frame_image,
                                                 left_eye_bbox[index, :])
                right_eye_image = self.crop_image(frame_image,
                                                  right_eye_bbox[index, :])

                # Save images
                Image.fromarray(face_image).save(
                    output_face_path.joinpath(f'{frame_index:05d}.jpg'),
                    quality=95)
                Image.fromarray(left_eye_image).save(
                    output_left_eye_path.joinpath(f'{frame_index:05d}.jpg'),
                    quality=95)
                Image.fromarray(right_eye_image).save(
                    output_right_eye_path.joinpath(f'{frame_index:05d}.jpg'),
                    quality=95)

                # Collect metadata
                metadata[md.RECORDING_INDEX] += [recording_index]
                metadata[md.FRAME_INDEX] += [frame_index]
                metadata[md.GAZE_X] += [dot_info['XCam'][index]]
                metadata[md.GAZE_Y] += [dot_info['YCam'][index]]
                metadata[md.FACE_GRID] += [face_grid_bbox[index,:]]

                print('Done: ', recording_index, frame_index)

        # Integrate
        metadata[md.RECORDING_INDEX] = np.stack(metadata[md.RECORDING_INDEX],
                                                axis=0).astype(np.int32)
        metadata[md.FRAME_INDEX] = np.stack(metadata[md.FRAME_INDEX],
                                            axis=0).astype(np.int32)
        metadata[md.GAZE_X] = np.stack(metadata[md.GAZE_X], axis=0)
        metadata[md.GAZE_Y] = np.stack(metadata[md.GAZE_X], axis=0)
        metadata[md.FACE_GRID] = np.stack(metadata[md.FACE_GRID], axis = 0).astype(np.uint8)

        metadata = pd.DataFrame(metadata)
        metadata = self.split_metadata(metadata)
        
        metadata.to_csv(self.output_path.joinpath(fs.METADATA_CSV))

    def read_json(self, filename):
        if not os.path.isfile(filename):
            self.log_error('Warning: No such file %s!' % filename)
            return None

        with open(filename) as f:
            try:
                data = json.load(f)
            except:
                data = None

        if data is None:
            self.log_error('Warning: Could not read file %s!' % filename)
            return None

        return data

    def prepare_path(self, path: Path, clear=False):
        if not path.is_dir():
            path.mkdir(parents=True)

        if clear:
            for f in path.glob('*'):
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    f.unlink()

        return path

    def log_error(self, msg, critical=False):
        print(msg)
        if critical:
            sys.exit(1)

    def crop_image(self, img, bbox):
        bbox = np.array(bbox, int)

        aSrc = np.maximum(bbox[:2], 0)
        bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

        aDst = aSrc - bbox[:2]
        bDst = aDst + (bSrc - aSrc)

        res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)
        res[aDst[1]:bDst[1], aDst[0]:bDst[0], :] = img[aSrc[1]:bSrc[1],
                                                       aSrc[0]:bSrc[0], :]

        return res

    def split_metadata(metadata: pd.DataFrame, ratio:float = 90) -> pd.DataFrame:
        randnums= np.random.randint(1, 100, metadata.shape[0])
        metadata[md.SPLIT] = [md.TRAIN if x < ratio else md.VALIDATE for x in randnums]
        return metadata
        

