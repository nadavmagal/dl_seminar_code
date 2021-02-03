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
from torch.utils.tensorboard import SummaryWriter

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP
from model import U3NETP

from model.unnet import create_unnet
from model.unnet_simple import create_unnet_simple

# ------- 1. define loss function --------
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

# model_name = 'u2net'  # 'u2netp'
# model_name = 'u2netp'
# model_name = 'un2etp_dyn'
# model_name = 'u3netp'
model_name = 'unnetp'
# model_name = 'unnetp_simple'

data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = r'../../../datasets/DUTS-TR/DUTS-TR-Image/'  # os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
tra_label_dir = r'../../../datasets/DUTS-TR/DUTS-TR-Mask/'  # os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

# model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
cur_date_time = time.strftime("%Y.%m.%d-%H.%M")
model_dir = os.path.join(f'../../../final_project_results/models_{model_name}/', cur_date_time) + os.sep
log_dir = os.path.join(r'../../../final_project_results/logs/', cur_date_time) + os.sep
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# initialize tensorboard
writer = SummaryWriter(log_dir)

epoch_num = 100000
batch_size_train = 1  # default 12
batch_size_val = 1
train_num = 0
val_num = 0
checkpoint_model_path = None
# checkpoint_model_path = r'/media/nadav/final_project_results/models/2021.01.25-20.47/u2netp_epoch_912_bce_itr_190960_train_0.27281275450844655_tar_0.024022218178619038.pth'

# tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)
tra_img_name_list = tra_img_name_list[:100]  # TODO: be careful and remove this

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    # tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)
    tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
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

# ------- 3. define model --------
# define the net
if (model_name == 'u2net'):
    net = U2NET(3, 1)
elif (model_name == 'u2netp'):
    net = U2NETP(3, 1)
elif (model_name == 'un2etp_dyn'):
    net = U2NETPDyn(3, 1)
elif (model_name == 'u3netp'):
    net = U3NETP(3, 1)
elif (model_name == 'unnetp'):
    net = create_unnet(3)
elif (model_name == 'unnetp_simple'):
    net = create_unnet_simple(3)

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

for epoch in range(start_epoch, epoch_num):
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

        if ite_num % 500 == 0:
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

    writer.add_scalars('loss_tar', {'train_tar': running_tar_loss / ite_num4val}, epoch + 1)
    writer.add_scalars('loss', {'train': running_loss / ite_num4val}, epoch + 1)
    if epoch % 3 == 0:
        cur_save_model_full_path = model_dir + model_name + f"_epoch_{epoch}_bce_itr_{ite_num}_train_{running_loss / ite_num4val}_tar_{running_tar_loss / ite_num4val}.pth"
        # torch.save(net.state_dict(), cur_save_model_full_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / ite_num4val,
        }, cur_save_model_full_path)
    # torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
    # ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
    running_loss = 0.0
    running_tar_loss = 0.0
    net.train()  # resume train
    ite_num4val = 0

# tensorboard --logdir=/media/nadav/final_project_results/logs/
