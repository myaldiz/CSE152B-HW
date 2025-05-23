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
from tqdm import tqdm
import model
import utils
from pathlib import Path
import scipy.io as io
# DATABASE = 'data'
from util_tools import worker_init_fn, datasize, getWriterPath

class Train_model_frontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """
    def __init__(self, opt):
        self.opt = opt
        self.tensorboard_interval = 100
        self.save_interval = 1000
        self.validation_interval = 200
        self.validation_size = 5
        self._eval = False
        self.device = 'cpu'
        print(opt)
        pass
    
    @property
    def writer(self):
        # print("get writer")
        return self._writer
    
    @writer.setter
    def writer(self, writer):
        print("set writer")
        self._writer = writer

    @property
    def train_loader(self):
        # print("get dataloader")
        return self._train_loader

    @train_loader.setter
    def train_loader(self, loader):
        print("set train loader")
        self._train_loader = loader

    def dataParallel(self):
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        for e in list(self.nets):
            self.nets[e] = nn.DataParallel(self.nets[e])
            print("net parallel: ", e)
        # for e in list(self.optimizers):
        #     self.nets[e].nn.DataParallel(self.nets[e])
        #     print("load net: ", e)
        #     self.optimizers = self.adamOptim(self.net, lr=self.config['model']['learning_rate'])
        pass

    def eval(self):
        self._eval = True

    def test(self):
        print("use testing dataset")
        self._test = True

    def loadModel(self):
        pass
    def prepare_var(self, data_paral=True):
        opt = self.opt
        # path
        self.save_path = Path(opt.experiment)

        self.colormap = io.loadmat(opt.colormap )['cmap']
        # Initialize image batch
        imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, 300, 300) )
        labelBatch = Variable(torch.FloatTensor(opt.batchSize, opt.numClasses, 300, 300) )
        maskBatch = Variable(torch.FloatTensor(opt.batchSize, 1, 300, 300) )
        labelIndexBatch = Variable(torch.LongTensor(opt.batchSize, 1, 300, 300) )

        # Initialize network
        # if opt.isDilation:
        #     print("load dilation!")
        #     encoder = model.encoderDilation().cuda()
        #     decoder = model.decoderDilation().cuda()
        # elif opt.isSpp:
        #     print("load SPP: Pyramid Scene Parsing Network!")
        #     encoder = model.encoderSPP().cuda()
        #     decoder = model.decoderSPP().cuda()
        # else:
        #     print("load vanilla Unet!")
        #     encoder = model.encoder().cuda()
        #     decoder = model.decoder().cuda()

        # Initialize network
        if opt.isDilation:
            encoder = model.encoderDilation().cuda()
            decoder = model.decoderDilation().cuda()
        elif opt.isSpp:
            encoder = model.encoderDilation().cuda()
            decoder = model.decoderDilation(isSpp = True).cuda()
        else:
            encoder = model.encoder().cuda()
            decoder = model.decoder().cuda()
        if opt.loadParams:
            print("load pretrained weights to encoder!")
            from model import loadPretrainedWeight
            loadPretrainedWeight(encoder, isOutput=False)

        self.iteration = 0

        # Move network and containers to gpu
        if not opt.noCuda:
            self.device = 'cuda'
#             imBatch = imBatch.cuda(opt.gpuId )
#             labelBatch = labelBatch.cuda(opt.gpuId )
#             labelIndexBatch = labelIndexBatch.cuda(opt.gpuId )
#             maskBatch = maskBatch.cuda(opt.gpuId )
#             encoder = encoder.cuda(opt.gpuId )
#             decoder = decoder.cuda(opt.gpuId )
            imBatch = imBatch.to(self.device )
            labelBatch = labelBatch.to(self.device )
            labelIndexBatch = labelIndexBatch.to(self.device )
            maskBatch = maskBatch.to(self.device )
            encoder = encoder.to(self.device )
            decoder = decoder.to(self.device )

        # Initialize dataLoader
        segDataset = dataLoader.BatchLoader(
                imageRoot = opt.imageRoot,
                labelRoot = opt.labelRoot,
                fileList = opt.fileList,
                imWidth = opt.imWidth, imHeight = opt.imHeight
                )
        # segLoader = DataLoader(segDataset, batch_size=opt.batchSize, num_workers=1, shuffle=True )


        self.nets = {'encoder': encoder, 'decoder': decoder}
        if data_paral:
            self.dataParallel()

        if opt.optimFunc == 'SGD':
            optimizer_enc = optim.SGD(encoder.parameters(), lr=opt.initLR, momentum=0.9, weight_decay=5e-4 )
            optimizer_dec = optim.SGD(decoder.parameters(), lr=opt.initLR, momentum=0.9, weight_decay=5e-4 )
        elif opt.optimFunc == 'Adam':
            optimizer_enc = optim.Adam(encoder.parameters(), lr=opt.initLR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            optimizer_dec = optim.Adam(decoder.parameters(), lr=opt.initLR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        self.optimizers = {'encoder': optimizer_enc, 'decoder': optimizer_dec}
        
        if opt.pretrained is not None:
            self.loadPretrainedModel()

        self.dataBatches = {'imBatch': imBatch, 'labelBatch': labelBatch, 
                        'labelIndexBatch': labelIndexBatch, 'maskBatch': maskBatch}
        


    def train(self):
        opt = self.opt
        self.lossArr = []
        self.accuracyArr = []
        # epoch = opt.iterId
        self.accuracy = np.zeros(opt.numClasses, dtype=np.float32 )
        # for i, dataBatch in enumerate(segLoader ):
        for epoch in range(0, opt.nepoch ):
            self.epoch = epoch
            self.iteration += 1
            # trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
            self.confcounts = np.zeros( (opt.numClasses, opt.numClasses), dtype=np.int64 )        
            print('====== Training epoch %d...'%epoch)
            for i, dataBatch in tqdm(enumerate(self.train_loader)):
                self.train_val_sample(dataBatch)
                self.iteration += 1
                # def early_stop(iter_max):
                #     if self.iteration > iter_max: break
                #     pass

                if self._eval and self.iteration % self.validation_interval == 0:
                    print('====== Validating...')
                    self.confcounts = np.zeros( (opt.numClasses, opt.numClasses), dtype=np.int64 )        
                    for j, dataBatch in tqdm(enumerate(self.val_loader)):
                        self.train_val_sample(dataBatch, task='val')
                        if j > self.validation_size:
                            break

                if self.iteration >= opt.iterationEnd:
                    np.save('%s/loss.npy' % opt.experiment, np.array(lossArr ) )
                    np.save('%s/accuracy.npy' % opt.experiment, np.array(accuracyArr ) )
                    self.saveModel()
                    self.saveModel_inference()
                    # torch.save(net.state_dict(), '%s/netFinal_%d.pth' % (opt.experiment, epoch+1) )
                    break
                # save model for retraining
                if self.iteration % self.save_interval == 0:
                    self.saveModel()
                    self.saveModel_inference()
                    if self._test:
                        print('====== testing ======')
                        self.confcounts = np.zeros( (opt.numClasses, opt.numClasses), dtype=np.int64 )
                        for j, dataBatch in tqdm(enumerate(self.test_loader)):
                            self.train_val_sample(dataBatch, task='test')
                            if j > self.validation_size:
                                break
                pass

            pass

    def sort_dataBatch(self, dataBatch):
        def put_data_into_batch(data_dict, name_data, data_batch_dict, name_data_batch):
            data_batch = data_batch_dict[name_data_batch]
            data = data_dict[name_data]
            # data_batch.data.resize_(data.size() )
            # data_batch.data.copy_(data )
            data_batch = Variable(data)
            return data_batch

        # imBatch = self.dataBatches['imBatch']
        # # Read data
        names_data = ['im', 'label', 'labelIndex', 'mask']
        names_data_batch = ['imBatch', 'labelBatch', 'labelIndexBatch', 'maskBatch']
        for i in range(len(names_data)):
            dataBatch[names_data_batch[i]] = \
                put_data_into_batch(dataBatch, names_data[i], 
                                    self.dataBatches, names_data_batch[i])
        return dataBatch


    def train_val_sample(self, dataBatch, task='train'):
        opt = self.opt
        dataBatch = self.sort_dataBatch(dataBatch)
        self.images_dict, self.scalar_dict = {}, {}
        # im, label, labelIndex, mask = dataBatch_['im'], dataBatch['label'], dataBatch['labelIndex'], dataBatch['mask']

        imBatch = dataBatch['imBatch']
        labelBatch = dataBatch['labelBatch']
        labelIndexBatch = dataBatch['labelIndexBatch']
        maskBatch = dataBatch['maskBatch']
        # Train network
        for e in list(self.optimizers):
            self.optimizers[e].zero_grad()

        imBatch = imBatch.to(self.device)
        labelBatch = labelBatch.to(self.device)
        labelIndexBatch = labelIndexBatch.to(self.device)
        maskBatch = maskBatch.to(self.device)
        
        x1, x2, x3, x4, x5 = self.nets['encoder'](imBatch )
        pred = self.nets['decoder'](imBatch, x1, x2, x3, x4, x5 )
        # Compute mean IOU
        loss = torch.mean( pred * labelBatch )
        # print("pred: ", pred)
        # print("labelBatch: ", labelBatch)

        # if task=='train':
        loss.backward()
        for e in list(self.optimizers):
            self.optimizers[e].step()

        def get_accuracy_from_counfcounts(confcounts, numClasses):
            accuracy = np.zeros(numClasses, dtype=np.float32 )
            for n in range(0, numClasses ):
                rowSum = np.sum(confcounts[n, :] )
                colSum = np.sum(confcounts[:, n] )
                interSum = confcounts[n, n]
                accuracy[n] = float(100.0 * interSum) / max(float(rowSum + colSum - interSum ), 1e-5)
            meanAccuracy = np.mean(accuracy )
            return meanAccuracy
        

        # Output the log information
        self.lossArr.append(loss.cpu().data.item() )

        if self.iteration >= 1000:
            meanLoss = np.mean(np.array(self.lossArr[-1000:] ) )
        else:
            meanLoss = np.mean(np.array(self.lossArr[:] ) )

        ## add to tensorboard
        if self.iteration % self.tensorboard_interval == 0:
            print('== Adding to Tensorboard')
            # calculate accuracy 
            hist = utils.computeAccuracy(pred, labelIndexBatch, maskBatch )
            self.confcounts += hist
            # self.confcounts = hist
            accuracy = get_accuracy_from_counfcounts(self.confcounts, opt.numClasses )
            self.accuracyArr.append(accuracy)
            self.scalar_dict.update({'accuracy': accuracy})
            # print loss and accuracy
            print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f' % (self.epoch, self.iteration, self.lossArr[-1], meanLoss ) )
            print('Epoch %d iteration %d: Accuracy %.5f' % ( self.epoch, self.iteration, accuracy ) )
            # add images
            # vutils.save_image( imBatch.data , '%s/images_%d.png' % (opt.experiment, iteration ), padding=0, normalize = True)
            np2torch = lambda x: torch.tensor(x.copy())
            img_trans = lambda x: x.transpose([2,0,1])
            
            labelGt = utils.save_label(labelBatch.data[:1], maskBatch.data[:1], self.colormap, None, nrows=1, ncols=1 )
            labelGt = np2torch(img_trans(labelGt)[np.newaxis,...])
            labelPred = utils.save_label(-pred.data[:1], maskBatch.data[:1], self.colormap, None, nrows=1, ncols=1 )
            labelPred = np2torch(img_trans(labelPred)[np.newaxis,...])
            # print("labelGt: ", labelGt.shape)

            self.images_dict.update({'labelGt': labelGt, 'labelPred': labelPred})
            self.tb_images_dict(self.writer, self.images_dict, self.iteration, task=task)

        self.scalar_dict.update({'loss': loss})
        self.loss = loss
        self.tb_scalar_dict(self.writer, self.scalar_dict, self.iteration, task=task)

    def saveModel(self):
        # save checkpoint
        data = {}
        for e in list(self.nets):
            data['net_'+e+'_state_dict'] = self.nets[e].state_dict()
        for e in list(self.optimizers):
            data['optimizer_'+e+'_state_dict'] = self.optimizers[e].state_dict()
        data.update({'n_iter': self.iteration + 1,
                     'loss': self.loss,
                    })

        self.save_checkpoint(self.save_path, data, self.iteration)
        print('== Model saved to ', self.save_path)
        pass
    
    def loadPretrainedModel(self):
        path = self.opt.pretrained
        print("load from: ", path)
        checkpoint = torch.load(path)
        for e in list(self.nets):
            # data['net_'+e+'_state_dict'] = self.nets[e].state_dict()
            self.nets[e].load_state_dict(checkpoint['net_'+e+'_state_dict'])
            print("load net: ", e)
        for e in list(self.optimizers):
            # data['optimizer_'+e+'_state_dict'] = self.optimizers[e].state_dict()
            self.optimizers[e].load_state_dict(checkpoint['optimizer_'+e+'_state_dict'])
            print("load optimizers: ", e)
        self.iteration = checkpoint['n_iter']

    def saveModel_inference(self):
        for e in list(self.nets):
            model = self.nets[e]
            torch.save(model.state_dict(), ('%s/'+e+'_%d.pth') % (str(self.save_path), self.iteration))
        pass

    @staticmethod
    def save_checkpoint(save_path, net_state, epoch, filename='checkpoint.pth.tar'):
        file_prefix = ['unet']
        # torch.save(net_state, save_path)
        filename = '{}_{}_{}'.format(file_prefix[0], str(epoch), filename)
        torch.save(net_state, save_path/filename)
        print("save checkpoint to ", filename)
        pass

    # from utils.utils import tb_scalar_dict
    @staticmethod
    def tb_scalar_dict(writer, scalar_dict, iter, task='training'):
        for element in list(scalar_dict):
            obj = scalar_dict[element]
            writer.add_scalar(task + '-' + element, obj, iter)

    @staticmethod
    def tb_images_dict(writer, tb_imgs, iter, task='training', max_img=5):
        for element in list(tb_imgs):
            for idx in range(tb_imgs[element].shape[0]):
                if idx >= max_img: break
                writer.add_image(task + '-' + element + '/%d'%idx, 
                    tb_imgs[element][idx,...], iter)

def main():
    parser = argparse.ArgumentParser()
    # The locationi of training set
    parser.add_argument('--imageRoot', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/JPEGImages', help='path to input images' )
    parser.add_argument('--labelRoot', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/SegmentationClass', help='path to input images' )
    parser.add_argument('--fileList', default='/datasets/cs152b-sp22-a00-public/hw4_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', help='path to input images' )
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
    parser.add_argument('--initLR', type=float, default=0.1, help='the initial learning rate')
    parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training')
    parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network')
    parser.add_argument('--iterationDecreaseLR', type=int, nargs='+', default=[16000, 24000], help='the iteration to decrease learning rate')
    parser.add_argument('--iterationEnd', type=int, default=28000, help='the iteration to end training')
    parser.add_argument('--optimFunc', default='SGD', help='select the optmizer')
    # add
    parser.add_argument('--loadParams', action='store_true', help='load pretrained parameters from resnet')
    parser.add_argument('--modelRoot', default='checkpoint', help='the path to store the testing results')
    parser.add_argument('--pretrained', default=None, help='file of checkpoint: checkpoint/*.tar')


    opt = parser.parse_args()
    print(opt)

    writer = SummaryWriter(getWriterPath(task=opt.experiment, date=True))
    segDataset = dataLoader.BatchLoader(
            imageRoot = opt.imageRoot,
            labelRoot = opt.labelRoot,
            fileList = opt.fileList,
            imWidth = opt.imWidth, imHeight = opt.imHeight
            )

    
    segLoader = DataLoader(segDataset, batch_size=opt.batchSize, num_workers=1, shuffle=True,
                            worker_init_fn=worker_init_fn)

    datasize(segLoader, opt.batchSize, tag='train')

    train_agent = Train_model_frontend(opt)
    train_agent.loadModel()
    train_agent.prepare_var()
    train_agent.train_loader = segLoader
    train_agent.writer = writer
    train_agent.train()

if __name__ == "__main__":
    main()

    pass
