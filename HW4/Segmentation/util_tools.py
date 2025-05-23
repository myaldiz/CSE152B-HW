
# import logging
# from utils.utils import datasize
def datasize(train_loader, batch_size, tag='train'):
    print('== %s split size %d in %d batches'%\
    (tag, len(train_loader)*batch_size, len(train_loader)))
    pass



def getWriterPath(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return prefix + task + '/' + exper_name + str_date_time

# from util_tools.utils import worker_init_fn
def worker_init_fn(worker_id):
   """The function is designed for pytorch multi-process dataloader.
   Note that we use the pytorch random generator to generate a base_seed.
   Please try to be consistent.
   References:
       https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
   """
   base_seed = torch.IntTensor(1).random_().item()
   # print(worker_id, base_seed)
   np.random.seed(base_seed + worker_id)

# from utils.utils import tb_scalar_dict
def tb_scalar_dict(writer, scalar_dict, iter, task='training'):
    for element in list(scalar_dict):
        obj = scalar_dict[element]
        writer.add_scalar(task + '-' + element, obj, iter)


import torch
# save model
# from utils.utils import save_model
def save_model(save_path, iter, net, optimizer, loss):
    torch.save(
        {
            'iter': iter,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        save_path)
    return True

# load model
def load_checkpoint(PATH):
    checkpoint = torch.load(PATH)
    return checkpoint
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

###########################################################
###################### PRINT    ###########################
###########################################################

# from util_tools.utils import datasize
def datasize(train_loader, batch_size, tag='train'):
    # import logging
    print('== %s split size %d in %d batches'%\
    (tag, len(train_loader)*batch_size, len(train_loader)))
    pass

# from util_tools.utils import print_dict_attr
def print_dict_attr(dictionary, attr=None, file=None):
    for item in list(dictionary):
        d = dictionary[item]
        if attr == None:
            print(item, ": ", d, file=file)
        else:
            if hasattr(d, attr):
                print(item, ": ", getattr(d, attr), file=file)
            else:
                print(item, ": ", len(d), file=file)

import numpy as np
# from util_tools.draw import img_overlap
def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    if img_gray.shape[0] == 1:
        img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img

###########################################################
###################### PRINT end ##########################
###########################################################
