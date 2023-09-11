#CUDA_VISIBLE_DEVICES=X python train.py --cuda --outpath ./outputs
from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from net import NetS, NetC
# from LoadData import Dataset, loader, Dataset_val
from LoadDataTiff import Dataset, loader, Dataset_val
import cv2

# Training settings
parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--batchSize', type=int, default=36, help='training batch size')
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.02')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use, for now it only supports one GPU')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
parser.add_argument('--cuda', action='store_true', help='using GPU or not')
parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=1111')
parser.add_argument('--outpath', default='./outputs', help='folder to output images and model checkpoints')
opt = parser.parse_args()

print(opt)


try:
    os.makedirs(opt.outpath)
except OSError:
    pass

# custom weights initialization called on NetS and NetC
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(input,target):
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    dice=2*(num/(den1+den2))

    dice_total=1-1*torch.sum(dice)/dice.size(0)#divide by batchsize

    return dice_total



cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True
print('===> Building model')
NetS = NetS(ngpu = opt.ngpu)
# NetS.apply(weights_init)
print(NetS)
NetC = NetC(ngpu = opt.ngpu)
# NetC.apply(weights_init)
print(NetC)

if cuda:
    NetS = NetS.cuda()
    NetC = NetC.cuda()
    # criterion = criterion.cuda()

# setup optimizer
lr = opt.lr
decay = opt.decay
optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999))
# load training data
dataloader = loader(Dataset('./'),opt.batchSize)
print('dataloader:', len(dataloader))
# load testing data
dataloader_val = loader(Dataset_val('./'), 36, shuffle=False)


max_iou = 0
NetS.train()
for epoch in range(opt.niter):
    dices, G_losses, D_losses = [], [], []
    for i, data in enumerate(dataloader, 1):
        #train C
        NetC.zero_grad()
        input, label = Variable(data[0]), Variable(data[1])
        if cuda:
            input = input.cuda()
            target = label.cuda()
        target = target.type(torch.FloatTensor)
        target = target.cuda()

        output = NetS(input)
        #output = F.sigmoid(output*k)
        output = F.sigmoid(output)
        output = output.detach()
        output_masked = input.clone()
        input_mask = input.clone()

        #detach G from the network
        for d in range(output_masked.shape[1]):
            output_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * output).squeeze(1)
        if cuda:
            output_masked = output_masked.cuda()
        result = NetC(output_masked)
        target_masked = input.clone()
        for d in range(output_masked.shape[1]):
            target_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * target).squeeze(1)
        if cuda:
            target_masked = target_masked.cuda()
        target_D = NetC(target_masked)
        loss_D = - torch.mean(torch.abs(result - target_D))
        loss_D.backward()
        optimizerD.step()
        
        #clip parameters in D
        for p in NetC.parameters():
            p.data.clamp_(-0.05, 0.05)

        #train G
        NetS.zero_grad()
        output = NetS(input)
        output = F.sigmoid(output)

        for d in range(output_masked.shape[1]):
            output_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * output).squeeze(1)
        if cuda:
            output_masked = output_masked.cuda()
        result = NetC(output_masked)
        for d in range(output_masked.shape[1]):
            target_masked[:,d,:,:] = (input_mask[:,d,:,:].unsqueeze(1) * target).squeeze(1)
        if cuda:
            target_masked = target_masked.cuda()
        target_G = NetC(target_masked)
        loss_dice = dice_loss(output,target)
        loss_G = torch.mean(torch.abs(result - target_G))
        loss_G_joint = torch.mean(torch.abs(result - target_G)) + loss_dice
        loss_G_joint.backward()
        optimizerG.step()
        dices.append(1 - loss_dice.item())
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())
        # break

    mdice = np.nanmean(dices, axis=0)
    mG_Loss = np.nanmean(G_losses, axis=0)
    mD_Loss = np.nanmean(D_losses, axis=0)
    print("===> Epoch[{}]({}/{}): Epoch Dice: {:.4f}".format(epoch, i, len(dataloader), mdice))
    print("===> Epoch[{}]({}/{}): Epoch G_Loss: {:.4f}".format(epoch, i, len(dataloader), mG_Loss))
    print("===> Epoch[{}]({}/{}): Epoch D_Loss: {:.4f}".format(epoch, i, len(dataloader), mD_Loss))

    
    if epoch % 1 == 0:
        NetS.eval()
        IoUs, dices = [], []
        
        for i, data_val in enumerate(dataloader_val, 1):
            input, gt = Variable(data_val[0]), Variable(data_val[1])
            fnames = data_val[2]
            if cuda:
                input = input.cuda()
                gt = gt.cuda()
            pred = NetS(input)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.type(torch.LongTensor)
            pred_np = pred.data.cpu().numpy()
            gt = gt.data.cpu().numpy()
            for x in range(input.size()[0]):
                IoU = np.sum(pred_np[x][gt[x]==1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x]==1]))
                dice = np.sum(pred_np[x][gt[x]==1])*2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))
                IoUs.append(IoU)
                dices.append(dice)

        NetS.train()
        IoUs = np.array(IoUs, dtype=np.float64)
        dices = np.array(dices, dtype=np.float64)
        mIoU = np.nanmean(IoUs, axis=0)
        mdice = np.nanmean(dices, axis=0)
        print('mIoU: {:.4f}'.format(mIoU))
        print('Dice: {:.4f}'.format(mdice))
        if mIoU > max_iou:
            max_iou = mIoU
            torch.save(NetS.state_dict(), '%s/NetS_epoch_%d_mIoU_%.4f_Dice_%.4f.pth' % (opt.outpath, epoch, mIoU, mdice))
            
            # save training images
            vutils.save_image(data[0],
            '%s/input_train_epoch_%d.png' % (opt.outpath, epoch),
            normalize=True)
            vutils.save_image(data[1].float(),
                    '%s/label_train_epoch_%d.png' % (opt.outpath, epoch),
                    normalize=True)
            vutils.save_image(output.data,
                    '%s/result_train_epoch_%d_Dice_%.4f.png' % (opt.outpath, epoch, mdice),
                    normalize=True)

            # save val images
            vutils.save_image(data_val[0],
                    '%s/input_val_epoch_%d.png' % (opt.outpath, epoch),
                    normalize=True)
            vutils.save_image(data_val[1].float(),
                    '%s/label_val_epoch_%d.png' % (opt.outpath, epoch),
                    normalize=True)
            pred = pred.type(torch.FloatTensor)
            vutils.save_image(pred.data,
                '%s/result_val_epoch_%d_Dice_%.4f.png' % (opt.outpath, epoch, mdice),
                normalize=True)

    if epoch % 25 == 0:
        lr = lr*decay
        if lr <= 0.00000001:
            lr = 0.00000001
        print('Learning Rate: {:.6f}'.format(lr))
        # print('K: {:.4f}'.format(k))
        print('Max mIoU: {:.4f}'.format(max_iou))
        optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
        optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999))
