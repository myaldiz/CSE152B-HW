import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import model
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--imageRoot', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
parser.add_argument('--labelRoot', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
parser.add_argument('--fileList', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', help='path to input images' )
parser.add_argument('--experiment', default='checkpoint', help='the path to store sampled images and models' )
parser.add_argument('--imHeight', type=int, default=320, help='height of input image' )
parser.add_argument('--imWidth', type=int, default=320, help='width of input image' )
parser.add_argument('--batchSize', type=int, default=16, help='the size of a batch' )
parser.add_argument('--nepoch', type=int, default=300, help='the training epoch' )
parser.add_argument('--numClasses', type=int, default=21, help='the number of classes' )
parser.add_argument('--isPretrained', default=True, help='whether to load the pretrained model or not' )
parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not' )
parser.add_argument('--isSpp', action='store_true', help='whether to do spatial pyramid or not' )
parser.add_argument('--initLREncoder', type=float, default=1e-5, help='the initial learning rate for encoder' )
parser.add_argument('--initLRDecoder', type=float, default=1e-4, help='the Initial learning rate for decoder' )
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training' )
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network' )
parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')

# The detail network setting
opt = parser.parse_args()
print(opt)

colormap = io.loadmat(opt.colormap )['cmap']

if opt.isSpp == True :
    opt.isDilation = False

if opt.isDilation:
    opt.experiment += '_dilation'
if opt.isSpp:
    opt.experiment += '_spp'

# Save all the codes
os.system('mkdir -p %s' % opt.experiment )
os.system('cp *.py %s' % opt.experiment )

writer = SummaryWriter(opt.experiment, flush_secs=1)


if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initialize image batch
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, opt.imHeight, opt.imWidth) )
labelBatch = Variable(torch.FloatTensor(opt.batchSize, opt.numClasses, opt.imHeight, opt.imWidth) )
maskBatch = Variable(torch.FloatTensor(opt.batchSize, 1, opt.imHeight, opt.imWidth ) )
labelIndexBatch = Variable(torch.LongTensor(opt.batchSize, 1, opt.imHeight, opt.imWidth ) )

# Initialize network
if opt.isDilation:
    encoder = model.encoderDilation()
    decoder = model.decoderDilation()
    # decoder = model.decoder()
elif opt.isSpp:
    encoder = model.encoderDilation()
    decoder = model.decoderDilation(isSpp = True)
else:
    encoder = model.encoder()
    decoder = model.decoder()
if opt.isPretrained:
    model.loadPretrainedWeight(encoder, isOutput = True )


# Move network and containers to gpu
if not opt.noCuda:
    imBatch = imBatch.cuda(opt.gpuId )
    labelBatch = labelBatch.cuda(opt.gpuId )
    labelIndexBatch = labelIndexBatch.cuda(opt.gpuId )
    maskBatch = maskBatch.cuda(opt.gpuId )
    encoder = encoder.cuda(opt.gpuId )
    decoder = decoder.cuda(opt.gpuId )


# Initialize optimizer
optEncoder = optim.Adam(encoder.parameters(), lr=opt.initLREncoder, betas=(0.5, 0.999) )
optDecoder = optim.Adam(decoder.parameters(), lr=opt.initLRDecoder, betas=(0.5, 0.999) )

# Initialize dataLoader
segDataset = dataLoader.BatchLoader(
        imageRoot = opt.imageRoot,
        labelRoot = opt.labelRoot,
        fileList = opt.fileList,
        imWidth = opt.imWidth,
        imHeight = opt.imHeight
        )
segLoader = DataLoader(segDataset,
        batch_size=opt.batchSize,
        num_workers=4,
        shuffle=True)

lossArr = []
accuracyArr = []
iteration = 0

print('====segLoader: batches in an epoch', len(segLoader))
for epoch in range(0, opt.nepoch ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    np.random.seed(epoch )
    for i, dataBatch in enumerate(segLoader ):
        iteration += 1

        # Read data
        with torch.no_grad():
            image_cpu = dataBatch['im']
            imBatch.resize_(image_cpu.size() )
            imBatch.data.copy_(image_cpu )

            label_cpu = dataBatch['label']
            labelBatch.resize_(label_cpu.size() )
            labelBatch.data.copy_(label_cpu )

            labelIndex_cpu = dataBatch['labelIndex' ]
            labelIndexBatch.resize_(labelIndex_cpu.size() )
            labelIndexBatch.data.copy_(labelIndex_cpu )

            mask_cpu = dataBatch['mask' ]
            maskBatch.resize_( mask_cpu.size() )
            maskBatch.data.copy_( mask_cpu )


        # Train network
        optEncoder.zero_grad()
        optDecoder.zero_grad()

        x1, x2, x3, x4, x5 = encoder(imBatch )
        pred = decoder(imBatch, x1, x2, x3, x4, x5)

        loss = torch.mean( pred * labelBatch )
        loss.backward()

        optEncoder.step()
        optDecoder.step()

        confcounts = utils.computeAccuracy(pred, labelIndexBatch, maskBatch )
        accuracy = np.zeros(opt.numClasses, dtype=np.float32 )
        for n in range(0, opt.numClasses ):
            rowSum = np.sum(confcounts[n, :] )
            colSum = np.sum(confcounts[:, n] )
            interSum = confcounts[n, n]
            accuracy[n] = float(100.0 * interSum) / max(float(rowSum + colSum - interSum ), 1e-5 )

        # Output the log information
        lossArr.append(loss.cpu().data.item() )
        accuracyArr.append(np.mean(accuracy ) )

        if iteration >= 1000:
            meanLoss = np.mean(np.array(lossArr[-1000:] ) )
            meanAccuracy = np.mean(np.array(accuracyArr[-1000:] ) )
        else:
            meanLoss = np.mean(np.array(lossArr[:] ) )
            meanAccuracy = np.mean(np.array(accuracyArr[:] ) )

        print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f'  \
                % (epoch, iteration, lossArr[-1], meanLoss ) )
        print('Epoch %d iteraion %d: Accuracy %.5f Accumulated Accuracy %.5f'  \
                % (epoch, iteration, accuracyArr[-1], meanAccuracy ) )
        trainingLog.write('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f \n' \
                % (epoch, iteration, lossArr[-1], meanLoss ) )
        trainingLog.write('Epoch %d iteration %d: Accuracy %.5f Accumulated Accuracy %.5f \n' \
                % (epoch, iteration, accuracyArr[-1], meanAccuracy ) )

        writer.add_scalar('train/loss', lossArr[-1], iteration)
        writer.add_scalar('train/accuracy', accuracyArr[-1], iteration)
        writer.flush()

        if iteration % 500 == 0 or iteration == 1:
            vutils.save_image( imBatch.data , '%s/images_%d.png' % (opt.experiment, iteration), padding=0, nrow=4, normalize = True)
            utils.save_label(labelBatch.data, maskBatch.data, colormap, '%s/labelGt_%d.png' % (opt.experiment, iteration ), nrows=2, ncols=4 )
            utils.save_label(-pred.data, maskBatch.data, colormap, '%s/labelPred_%d.png' % (opt.experiment, iteration), nrows=2, ncols=4 )

    trainingLog.close()

    if epoch % 30 == 0:
        np.save('%s/loss.npy' % opt.experiment, np.array(lossArr ) )
        np.save('%s/accuracy.npy' % opt.experiment, np.array(accuracyArr ) )
        torch.save( encoder.state_dict(), '%s/encoder_%d.pth' % (opt.experiment, epoch+1) )
        torch.save( decoder.state_dict(), '%s/decoder_%d.pth' % (opt.experiment, epoch+1) )

    if (epoch+1) % 30 == 0:
        for param_group in optEncoder.param_groups:
            param_group['lr'] /= 2
        for param_group in optDecoder.param_groups:
            param_group['lr'] /= 2
