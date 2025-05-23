import torch
torch.manual_seed(0)
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import model
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--imageRoot', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
parser.add_argument('--labelRoot', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
parser.add_argument('--fileList', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', help='path to input images' )
parser.add_argument('--experiment', default='test', help='the path to store sampled images and models' )
parser.add_argument('--modelRoot', default='checkpoint', help='the path to store the testing results')
parser.add_argument('--epochId', type=int, default=300, help='the number of epochs being trained')
parser.add_argument('--batchSize', type=int, default=16, help='the size of a batch' )
parser.add_argument('--numClasses', type=int, default=21, help='the number of classes' )
parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not' )
parser.add_argument('--isSpp', action='store_true', help='whether to do spatial pyramid or not' )
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training' )
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network' )
parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')

# The detail network setting
opt = parser.parse_args()
print(opt)

colormap = io.loadmat(opt.colormap )['cmap']

#assert(opt.batchSize == 1 )

if opt.isSpp == True :
    opt.isDilation = False

if opt.isDilation:
    opt.experiment += '_dilation'
    opt.modelRoot += '_dilation'
if opt.isSpp:
    opt.experiment += '_spp'
    opt.modelRoot += '_spp'

# Save all the codes
os.system('mkdir -p %s' % opt.experiment )
os.system('cp *.py %s' % opt.experiment )

if torch.cuda.is_available() and opt.noCuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Initialize image batch
imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, 300, 300) )
labelBatch = Variable(torch.FloatTensor(opt.batchSize, opt.numClasses, 300, 300) )
maskBatch = Variable(torch.FloatTensor(opt.batchSize, 1, 300, 300) )
labelIndexBatch = Variable(torch.LongTensor(opt.batchSize, 1, 300, 300) )

# Initialize network
if opt.isDilation:
    encoder = model.encoderDilation()
    decoder = model.decoderDilation()
elif opt.isSpp:
    encoder = model.encoderDilation()
    decoder = model.decoderDilation(isSpp = True)
else:
    encoder = model.encoder()
    decoder = model.decoder()

encoder.load_state_dict(torch.load('%s/encoder_%d.pth' % (opt.modelRoot, opt.epochId) ) )
decoder.load_state_dict(torch.load('%s/decoder_%d.pth' % (opt.modelRoot, opt.epochId) ) )
encoder = encoder.eval()
decoder = decoder.eval()

# Move network and containers to gpu
if not opt.noCuda:
    imBatch = imBatch.cuda(opt.gpuId )
    labelBatch = labelBatch.cuda(opt.gpuId )
    labelIndexBatch = labelIndexBatch.cuda(opt.gpuId )
    maskBatch = maskBatch.cuda(opt.gpuId )
    encoder = encoder.cuda(opt.gpuId )
    decoder = decoder.cuda(opt.gpuId )

# Initialize dataLoader
segDataset = dataLoader.BatchLoader(
        imageRoot = opt.imageRoot,
        labelRoot = opt.labelRoot,
        fileList = opt.fileList,
        imHeight = 320,
        imWidth = 320
        )
segLoader = DataLoader(segDataset, batch_size=opt.batchSize, num_workers=0, shuffle=False )

lossArr = []
iteration = 0
epoch = opt.epochId
confcounts = np.zeros( (opt.numClasses, opt.numClasses), dtype=np.int64 )
accuracy = np.zeros(opt.numClasses, dtype=np.float32 )
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
for i, dataBatch in enumerate(segLoader ):
    iteration += 1

    # Read data
    label_name = dataBatch['labelName']
    print([_.split('/')[-1] for _ in label_name])
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

    # Test network
    x1, x2, x3, x4, x5 = encoder(imBatch )
    pred = decoder(imBatch, x1, x2, x3, x4, x5 )

    loss = torch.mean( pred * labelBatch )
    hist = utils.computeAccuracy(pred, labelIndexBatch, maskBatch )
    confcounts += hist

    for n in range(0, opt.numClasses ):
        rowSum = np.sum(confcounts[n, :] )
        colSum = np.sum(confcounts[:, n] )
        interSum = confcounts[n, n]
        accuracy[n] = float(100.0 * interSum) / max(float(rowSum + colSum - interSum ), 1e-5)

    # Output the log information
    lossArr.append(loss.cpu().data.item() )
    meanLoss = np.mean(np.array(lossArr[:] ) )
    meanAccuracy = np.mean(accuracy )

    print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f'  \
            % ( epoch, iteration, lossArr[-1], meanLoss ) )
    print('Epoch %d iteration %d: Accumulated Accuracy %.5f' \
            % ( epoch, iteration, meanAccuracy ) )
    testingLog.write('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f \n' \
            % ( epoch, iteration, lossArr[-1], meanLoss ) )
    testingLog.write('Epoch %d iteration %d: Accumulated Accuracy %.5f \n' \
            % ( epoch, iteration, meanAccuracy ) )

    if iteration % 50 == 0:
        vutils.save_image(imBatch.data , '%s/images_%d.png' % (opt.experiment, iteration ), padding=0, normalize = True)
        utils.save_label(labelBatch.data, maskBatch.data, colormap, '%s/labelGt_%d.png' % (opt.experiment, iteration ), nrows=1, ncols=1 )
        utils.save_label(-pred.data, maskBatch.data, colormap, '%s/labelPred_%d.png' % (opt.experiment, iteration ), nrows=1, ncols=1 )

testingLog.close()
np.save('%s/accuracy_%d.npy' % (opt.experiment, opt.epochId), accuracy )

print(accuracy)
