import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
from u2net_test import normPRED

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    # # loss0.data[0], loss1.data[0], loss2.data[0], loss3.data[0], loss4.data[0], loss5.data[0], loss6.data[0]))
    # loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))
    return loss0, loss


def validation_epoch(net, salobj_val_dataloader):
    net.eval()

    val_loss = 0.0
    val_tar_loss = 0.0

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(salobj_val_dataloader):

        inputs_test, labels_test = data_test['image'], data_test['label']
        inputs_test = inputs_test.type(torch.FloatTensor)
        labels_test = labels_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test, labels = Variable(inputs_test.cuda()), Variable(labels_test.cuda())
        else:
            inputs_test, labels = Variable(inputs_test), Variable(labels_test)

        d0, d1, d2, d3, d4, d5, d6 = net(inputs_test)

        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_test)

        val_loss += loss.item()
        val_tar_loss += loss2.item()

        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

    return val_loss, val_tar_loss
