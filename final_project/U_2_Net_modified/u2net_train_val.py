import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import time
import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from u2net_val import validation_epoch

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------

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


# ------- 2. set the directory of training dataset --------

model_name = 'u2netp'

data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = r'../../../datasets/DUTS-TR/DUTS-TR-Image/'  # os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
tra_label_dir = r'../../../datasets/DUTS-TR/DUTS-TR-Mask/'  # os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

# model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
cur_date_time = time.strftime("%Y.%m.%d-%H.%M")
model_dir = os.path.join(r'../../../final_project_results/models/', cur_date_time) + os.sep
os.makedirs(model_dir, exist_ok=True)

epoch_num = 100000
val_portion = 0.2
batch_size_train = 2
batch_size_val = 1  # might work only with 1
train_num = 0
val_num = 0
checkpoint_model_path = None
# checkpoint_model_path = r'/home/nadav/dl_seminar/final_project_results/models/2021.01.21-18.55/u2netp_ephoch_1_bce_itr_100_train_3.5880941772460937_tar_0.5038655388355255.pth'


# tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list[:100]:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    # tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)
    tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)

# separate to train and validation
val_img_num = int(val_portion * len(tra_img_name_list))
val_img_name_list = tra_img_name_list[:val_img_num]
val_lbl_name_list = tra_lbl_name_list[:val_img_num]
tra_img_name_list = tra_img_name_list[val_img_num:]
tra_lbl_name_list = tra_lbl_name_list[val_img_num:]

print("---")
print(f"train --> images: {len(tra_img_name_list)} | labels: {len(tra_lbl_name_list)}")
print(f"validation --> labels: images: {len(val_img_name_list)} | labels: {len(val_lbl_name_list)}")
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

# load validation data
salobj_val_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        ToTensorLab(flag=0)]))
salobj_val_dataloader = DataLoader(salobj_val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=1)

# ------- 3. define model --------
# define the net
if (model_name == 'u2net'):
    net = U2NET(3, 1)
elif (model_name == 'u2netp'):
    net = U2NETP(3, 1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
start_epoch = 0
if checkpoint_model_path is not None:
    checkpoint = torch.load(checkpoint_model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss'] + 1

print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000  # save the model every 2000 iterations
# save_frq = 5  # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        # running_loss += loss.data[0]
        running_loss += loss.item()
        # running_tar_loss += loss2.data[0]
        running_tar_loss += loss2.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        if ite_num % 20 == 0:
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

    # run validation
    val_loss, val_tar_loss = validation_epoch(net, salobj_val_dataloader)
    print("[epoch: %3d/%3d val loss: %3f, tar: %3f " % (epoch + 1, epoch_num, val_loss, val_tar_loss))

    if epoch % 10 == 0:
        cur_save_model_full_path = model_dir + model_name + f"_ephoch_{epoch}_bce_itr_{ite_num}_train_{running_loss / ite_num4val}_tar_{running_tar_loss / ite_num4val}.pth"
        # torch.save(net.state_dict(), cur_save_model_full_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / ite_num4val,
        }, cur_save_model_full_path)

    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
