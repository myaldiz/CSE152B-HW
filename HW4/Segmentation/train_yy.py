import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
from tensorboardX import SummaryWriter
from util_tools import getWriterPath, worker_init_fn, datasize
from tqdm import tqdm
import model
from pathlib import Path
from Train_model_frontend import Train_model_frontend
import pathlib


def main():
    parser = argparse.ArgumentParser()
    # The locationi of training set
    parser.add_argument('--imageRoot', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
    parser.add_argument('--labelRoot', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
    parser.add_argument('--fileList', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', help='list of training images' )
    # parser.add_argument('--fileListVal', default='/data/datasets/cs152b-sp22-a00-public/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt', help='list of validation images' )
    parser.add_argument('--fileListTest', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', help='list of validation images' )
    parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')

    parser.add_argument('--experiment', default='checkpoint', help='the path to store sampled images and models')
    # parser.add_argument('--marginFactor', type=float, default=4, help='margin factor')
    parser.add_argument('--imHeight', type=int, default=512, help='height of input image')
    parser.add_argument('--imWidth', type=int, default=512, help='width of input image') # according to https://github.com/meetshah1995/pytorch-semseg

    parser.add_argument('--numClasses', type=int, default=21, help='the number of classes' )
    parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not' )
    parser.add_argument('--isSpp', action='store_true', help='whether to do spatial pyramid or not' )

    parser.add_argument('--batchSize', type=int, default=8, help='the size of a batch')
    parser.add_argument('--nepoch', type=int, default=8, help='the training epoch')
    parser.add_argument('--num_workers', type=int, default=1, help='the training epoch')
    parser.add_argument('--initLR', type=float, default=0.1, help='the initial learning rate')
    parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training')
    parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network')
    parser.add_argument('--iterationDecreaseLR', type=int, nargs='+', default=[16000, 24000], help='the iteration to decrease learning rate')
    parser.add_argument('--iterationEnd', type=int, default=28000, help='the iteration to end training')
    parser.add_argument('--optimFunc', default='SGD', help='select the optmizer')

    parser.add_argument('--loadParams', action='store_true', help='load pretrained parameters from resnet')
    parser.add_argument('--modelRoot', default='checkpoint', help='the path to store the testing results')
    parser.add_argument('--pretrained', default=None, help='file of checkpoint: checkpoint/*.tar')


    opt = parser.parse_args()
    print(opt)

    if opt.isSpp == True :
        opt.isDilation = False

    if opt.isDilation:
        opt.experiment += '_dilation'
        opt.modelRoot += '_dilation'

    # init writer
    summary_path = getWriterPath(task=opt.experiment, date=True)
    print('== Summary path:', summary_path)
    writer = SummaryWriter(summary_path)

    # Save all the codes
    # os.system('mkdir %s' % opt.experiment )
    Path(opt.experiment).mkdir(parents=True, exist_ok=True)
    os.system('cp *.py %s' % opt.experiment )

    if torch.cuda.is_available() and opt.noCuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Initialize dataLoader
    segDataset = dataLoader.BatchLoader(
            imageRoot = opt.imageRoot,
            labelRoot = opt.labelRoot,
            fileList = opt.fileList,
            imWidth = opt.imWidth, imHeight = opt.imHeight
            )
    test_dataset = dataLoader.BatchLoader(
            imageRoot = opt.imageRoot,
            labelRoot = opt.labelRoot,
            fileList = opt.fileListTest,
            imWidth = opt.imWidth, imHeight = opt.imHeight
            )

    num_workers = opt.num_workers
    print(f"num_workers: {num_workers}")
    # segLoader = DataLoader(segDataset, batch_size=opt.batchSize, num_workers=num_workers, shuffle=True,
    #                         worker_init_fn=worker_init_fn)

    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, num_workers=num_workers, shuffle=True,
                            worker_init_fn=worker_init_fn)

    def split_dataset(dataset):
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        return train_dataset, val_dataset
    train_dataset, val_dataset = split_dataset(segDataset)

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, num_workers=num_workers, shuffle=True,
                            worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, num_workers=num_workers, shuffle=True,
                            worker_init_fn=worker_init_fn)

    datasize(train_loader, opt.batchSize, tag='train')
    datasize(val_loader, opt.batchSize, tag='val')
    datasize(test_loader, opt.batchSize, tag='test')

    train_agent = Train_model_frontend(opt)
    train_agent.loadModel()
    train_agent.prepare_var()
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader
    train_agent.test_loader = test_loader
    train_agent.eval()
    train_agent.test()
    train_agent.writer = writer
    train_agent.train()

    # Save the accuracy
    # np.save('%s/accuracy_%d.npy' % (opt.experiment, opt.epochId), accuracy )

if __name__ == "__main__":
    main()
