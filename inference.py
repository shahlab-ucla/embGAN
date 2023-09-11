from __future__ import print_function
import argparse
import os
from PIL import Image
import torch
from torch.utils import data
from torch.autograd import Variable
from net import NetS, NetC
from LoadDataTiff import Dataset, loader, Dataset_infer
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--out_path', default='')
parser.add_argument('--data_path', default='./')
parser.add_argument('--model_path')
parser.add_argument('--save_input', action='store_true')
parser.add_argument('--cuda', action='store_true', help='using GPU or not')
opt = parser.parse_args()

# cuda = opt.cuda
cuda = True
NetS = NetS(ngpu = 1)
NetS.load_state_dict(torch.load(opt.model_path))

if cuda:
    NetS = NetS.cuda()

# load data
dataloader_infer = loader(Dataset_infer(opt.data_path), 36, shuffle=False)

NetS.eval()

try:
    os.makedirs(opt.out_path)
except OSError:
    pass

for i, data in enumerate(dataloader_infer, 1):
    input = Variable(data[0])
    x = input.detach().cpu().numpy()
    fnames = data[1]
    if cuda:
        input = input.cuda()
    pred = NetS(input)

    # save probability maps
    for x in range(input.size()[0]):
        original_size = Image.open(fnames[x]).size
        fname = os.path.split(fnames[x])[1]

        # save prediction
        np_pred = pred[x].detach().cpu().numpy()
        im = Image.fromarray(np.squeeze(np_pred)).resize(original_size, Image.BILINEAR)
        im.save(os.path.join(opt.out_path, fname), mode='F')

        if opt.save_input:
            # save input image
            input_im = input[x].detach().cpu().numpy()
            im = Image.fromarray(np.squeeze(input_im)).resize(original_size, Image.BILINEAR)
            im.save(os.path.join(opt.out_path,'input_' + fname), mode='F')

