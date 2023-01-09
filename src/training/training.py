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

from .dataset import Dataset
from . import metadata as md

CHECKPOINTS_PATH = '.'


class Train:

    def __init__(self,
                 data_path: str,
                 epochs: int = 25,
                 lr: float = 0.0001,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 workers: int = 16,
                 reset: bool = False,
                 sink: bool = False) -> None:

        # Change there flags to control what happens.
        self.data_path = data_path
        # Do not load checkpoint at the beginning
        self.reset = reset
        # Only run test, no training
        self.sink = sink
        self.workers = workers
        self.epochs = epochs
        # Change if out of cuda memory
        self.batch_size = torch.cuda.device_count() * 100

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.print_freq = 10
        self.prec1 = 0

    def __call__(self, model: nn.Module) -> None:

        model = torch.nn.DataParallel(model)
        model.cuda()
        cudnn.benchmark = True

        # initialize learning params
        image_size = (224, 224)
        best_prec1 = 1e20
        lr = self.lr

        epoch = 0
        if not self.reset:
            saved = self.load_checkpoint()
            if saved:
                print(
                    'Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...'
                    % (saved['epoch'], saved['best_prec1']))
                state = saved['state_dict']
                try:
                    model.module.load_state_dict(state)
                except:
                    model.load_state_dict(state)
                epoch = saved['epoch']
                best_prec1 = saved['best_prec1']
            else:
                print('Warning: Could not read checkpoint!')

        data_train = Dataset(data_path=self.data_path,
                                split=md.TRAIN,
                                image_size=image_size)
        data_val = Dataset(data_path=self.data_path,
                              split=md.VALIDATE,
                              image_size=image_size)

        train_loader = torch.utils.data.DataLoader(data_train,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.workers,
                                                   pin_memory=True)

        val_loader = torch.utils.data.DataLoader(data_val,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.workers,
                                                 pin_memory=True)

        criterion = nn.MSELoss().cuda()

        # optimizer = torch.optim.Adam(model.parameters(),
                                    # lr,
                                    # weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr,
                                    momentum=self.momentum,
                                    nesterov=True,
                                    weight_decay=self.weight_decay)

        # Quick test
        if self.sink:
            self.validate(model, val_loader, criterion, epoch)
            return

        for epoch in range(0, epoch):
            lr = self.adjust_learning_rate(optimizer, epoch)

        for epoch in range(epoch, self.epochs):
            lr = self.adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            self.train(model, train_loader, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = self.validate(model, val_loader, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            self.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best)

    def train(self, model: nn.Module, train_loader, criterion, optimizer,
              epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossesLin = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()

        for i, (row, face, left_eye, right_eye, face_grid,
                gaze) in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)
            face = face.cuda()
            left_eye = left_eye.cuda()
            right_eye = right_eye.cuda()
            face_grid = face_grid.cuda()
            gaze = gaze.cuda()

            face = torch.autograd.Variable(face, requires_grad=True)
            left_eye = torch.autograd.Variable(left_eye, requires_grad=True)
            right_eye = torch.autograd.Variable(right_eye, requires_grad=True)
            face_grid = torch.autograd.Variable(face_grid, requires_grad=True)
            gaze = torch.autograd.Variable(gaze, requires_grad=False)

            # compute output
            output = model(face, left_eye, right_eye, face_grid)

            loss = criterion(output, gaze)

            losses.update(loss.data.item(), face.size(0))

            lossLin = output - gaze
            lossLin = torch.mul(lossLin, lossLin)
            lossLin = torch.sum(lossLin, 1)
            lossLin = torch.mean(torch.sqrt(lossLin))

            lossesLin.update(lossLin.item(), face.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      lossLin=lossesLin))

    def validate(self, model: nn.Module, val_loader, criterion, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        lossesLin = AverageMeter()

        # switch to evaluate mode
        model.eval()
        end = time.time()

        for i, (row, face, left_eye, right_eye, face_grid,
                gaze) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            face = face.cuda()
            left_eye = left_eye.cuda()
            right_eye = right_eye.cuda()
            face_grid = face_grid.cuda()
            gaze = gaze.cuda()

            face = torch.autograd.Variable(face, requires_grad=False)
            left_eye = torch.autograd.Variable(left_eye, requires_grad=False)
            right_eye = torch.autograd.Variable(right_eye, requires_grad=False)
            face_grid = torch.autograd.Variable(face_grid, requires_grad=False)
            gaze = torch.autograd.Variable(gaze, requires_grad=False)

            # compute output
            with torch.no_grad():
                output = model(face, left_eye, right_eye, face_grid)

            loss = criterion(output, gaze)

            lossLin = output - gaze
            lossLin = torch.mul(lossLin, lossLin)
            lossLin = torch.sum(lossLin, 1)
            lossLin = torch.mean(torch.sqrt(lossLin))

            losses.update(loss.data.item(), face.size(0))
            lossesLin.update(lossLin.item(), face.size(0))

            # compute gradient and do SGD step
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                      epoch,
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      lossLin=lossesLin))

        return lossesLin.avg

    def load_checkpoint(self, filename='checkpoint.pth.tar'):
        filename = os.path.join(CHECKPOINTS_PATH, filename)
        print(filename)
        if not os.path.isfile(filename):
            return None
        state = torch.load(filename)
        return state

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not os.path.isdir(CHECKPOINTS_PATH):
            os.makedirs(CHECKPOINTS_PATH, 0o777)
        bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
        filename = os.path.join(CHECKPOINTS_PATH, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, bestFilename)

    def adjust_learning_rate(self, optimizer, epoch) -> float:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.lr * (0.1**(epoch // 30))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr
        return lr

    def save_model(self, model: nn.Module, path: Path):
        saved = self.load_checkpoint()
        if saved:
            print(
                'Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...'
                % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            torch.save(model, path)
        else:
            raise Exception('Error: Could not read checkpoint!')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
