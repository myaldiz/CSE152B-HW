import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import faceNet
import faceNetBN
import torch.nn as nn
import os
import numpy as np
from pathlib import Path
import shutil

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
# The location of training set
parser.add_argument('--imageRoot',        default='../CASIA-WebFace/',               help='path to input images')
parser.add_argument('--alignmentRoot',    default='./data/casia_landmark.txt',       help='path to the alignment file')
parser.add_argument('--experiment',       default='checkpoint',                      help='the path to store sampled images and models')

# CosFace hyper-parameters (as per the paper)
parser.add_argument('--marginFactor',     type=float, default=0.35,                    help='margin factor (m)')
parser.add_argument('--scaleFactor',      type=float, default=30.0,                    help='scale factor (s)')

# Image and batch settings
parser.add_argument('--imHeight',         type=int,   default=112,                     help='height of input image')
parser.add_argument('--imWidth',          type=int,   default= 96,                     help='width of input image')
parser.add_argument('--batchSize',        type=int,   default=512,                     help='the size of a batch')             # ↑ increased batch size

# Training schedule
parser.add_argument('--nepoch',           type=int,   default=30,                      help='number of training epochs')
parser.add_argument('--initLR',           type=float, default=0.1,                     help='initial learning rate')           # ↑ set to 0.1
parser.add_argument(
    '--iterationDecreaseLR',
    type=int,
    nargs='+',
    default=[54124, 77320, 88918],                                             # ↑ decay at epochs 14, 20, 23
    help='the iterations at which to decrease the learning rate by 10×'
)
parser.add_argument('--iterationEnd',     type=int,   default=3866*30,                help='the iteration to end training')

# CUDA and logging
parser.add_argument('--noCuda',           action='store_true',                       help='do not use cuda for training')
parser.add_argument('--gpuId',            type=int,   default=0,                      help='gpu id used for training the network')

opt = parser.parse_args()
print(opt)

# prepare experiment directory
opt.experiment = str(Path('logs') / opt.experiment)
if Path(opt.experiment).exists():
    shutil.rmtree(opt.experiment, ignore_errors=True)
    print('Removed ' + str(opt.experiment))
os.makedirs(opt.experiment, exist_ok=True)

# Save all the codes
os.system(f'cp *.py {opt.experiment}')

writer = SummaryWriter(opt.experiment, flush_secs=10)
print('=====> Summary writing to %s' % opt.experiment)

if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run without --noCuda")

# Initialize network
net       = faceNetBN.faceNet(m=opt.marginFactor, feature=False)
lossLayer = faceNet.CustomLoss(s=opt.scaleFactor)

# Move to GPU
if not opt.noCuda:
    net = net.cuda(opt.gpuId)

# Optimizer: SGD with momentum and weight decay
optimizer = optim.SGD(net.parameters(), lr=opt.initLR, momentum=0.9, weight_decay=5e-4)

# DataLoader with shuffling enabled
faceDataset = dataLoader.BatchLoader(
    imageRoot     = opt.imageRoot,
    alignmentRoot = opt.alignmentRoot,
    cropSize      = (opt.imWidth, opt.imHeight)
)
faceLoader = DataLoader(
    faceDataset,
    batch_size = opt.batchSize,
    num_workers= 16,
    shuffle     = True                                       # ↑ enabled shuffle
)

lossArr      = []
accuracyArr  = []
iteration    = 0

for epoch in range(opt.nepoch):
    trainingLog = open(f'{opt.experiment}/trainingLog_{epoch}.txt', 'w')
    for i, dataBatch in enumerate(faceLoader):
        iteration += 1

        # Read data
        imBatch     = Variable(dataBatch['img'])
        targetBatch = Variable(dataBatch['target'])
        if not opt.noCuda:
            imBatch     = imBatch.cuda()
            targetBatch = targetBatch.cuda()

        # Train step
        optimizer.zero_grad()
        pred = net(imBatch)
        loss, accuracy = lossLayer(pred, targetBatch)
        loss.backward()
        optimizer.step()

        # Log scalars
        lossArr.append(loss.item())
        accuracyArr.append(accuracy)
        writer.add_scalar('loss_train/loss', loss.item(), iteration)
        writer.add_scalar('loss_train/accuracy', accuracy, iteration)
        writer.add_scalar('training/epoch', epoch, iteration)
        writer.add_scalar('training/iteration', iteration, iteration)
        writer.add_scalar('training/lr', optimizer.param_groups[0]['lr'], iteration)

        # Print & record running stats
        window = lossArr[-1000:] if iteration >= 1000 else lossArr
        meanLoss     = np.mean(window)
        meanAccuracy = np.mean(accuracyArr[-1000:] if iteration >= 1000 else accuracyArr)

        print(f'Epoch {epoch} iteration {iteration}: Loss {lossArr[-1]:.5f} Accum. Loss {meanLoss:.5f}')
        print(f'Epoch {epoch} iteration {iteration}: Acc  {accuracyArr[-1]:.5f} Accum. Acc {meanAccuracy:.5f}')
        trainingLog.write(
            f'Epoch {epoch} iteration {iteration}: Loss {lossArr[-1]:.5f} Accum. Loss {meanLoss:.5f}\n'
            f'Epoch {epoch} iteration {iteration}: Acc  {accuracyArr[-1]:.5f} Accum. Acc {meanAccuracy:.5f}\n'
        )

        # Save first batch images
        if iteration == 1:
            vutils.save_image(0.5 * (imBatch.data + 1), f'{opt.experiment}/images.png')

        # LR decay at specified iterations
        if iteration in opt.iterationDecreaseLR:
            print(f'Learning rate decreased at iteration {iteration}')
            trainingLog.write(f'Learning rate decreased at iteration {iteration}\n')
            for pg in optimizer.param_groups:
                pg['lr'] /= 10

        # Final checkpoint
        if iteration >= opt.iterationEnd:
            np.save(f'{opt.experiment}/loss.npy', np.array(lossArr))
            np.save(f'{opt.experiment}/accuracy.npy', np.array(accuracyArr))
            torch.save(net.state_dict(), f'{opt.experiment}/netFinal_{epoch+1}.pth')
            break

    trainingLog.close()
    if iteration >= opt.iterationEnd:
        break

    # Periodic snapshot every 2 epochs
    if (epoch + 1) % 2 == 0:
        np.save(f'{opt.experiment}/loss.npy', np.array(lossArr))
        np.save(f'{opt.experiment}/accuracy.npy', np.array(accuracyArr))
        torch.save(net.state_dict(), f'{opt.experiment}/net_{epoch+1}.pth')
