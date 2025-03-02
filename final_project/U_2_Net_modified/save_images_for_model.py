import os
from skimage import io, transform
import torch
import time
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import cv2

import json
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from model import U3NETP

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name,pred,output_dir_w_time):
    test_label_full_path = r'../../..//datasets/DUTS-TE/DUTS-TE-Mask'
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np_255 = (255*predict_np).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir_w_time, os.path.basename(image_name)[:-4] + '.png'), predict_np_255)


def save_output_original(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')




def main():

    # --------- 1. get image path and name ---------
    # model_name='u2net'#u2netp
    # model_name= 'u2netp'
    model_name= 'u3netp'
    model_dir = r'/media/nadav/final_project_results/models_u3netp/val_2021.02.16-22.58/u3netp_ephoch_1002_bce_itr_35200_train_0.23393689743137325_tar_0.018960884751074693_valloss_6.688648055209393.pth'

    # RUN_ON_GPU = False
    RUN_ON_GPU = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    image_dir = r'../../../datasets/DUTS-TE/DUTS-TE-Image/'

    output_dir = f'../../../final_project_results/test_pred_mask/{os.path.basename(model_dir)[:-4]}/'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")
    output_dir_w_time = os.path.join(output_dir, cur_date_time) + os.sep
    if not os.path.exists(output_dir_w_time):
        os.makedirs(output_dir_w_time, exist_ok=True)

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    elif (model_name=='u3netp'):
        net = U3NETP(3,1)

    # net.load_state_dict(torch.load(c))
    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available() and RUN_ON_GPU:
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    all_images_mae = []
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
        print(f'{i_test}/{len(img_name_list)}')

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available() and RUN_ON_GPU:
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        # save_output_original(img_name_list[i_test],pred,output_dir_w_time)
        save_output(img_name_list[i_test],pred,output_dir_w_time)

        del d1,d2,d3,d4,d5,d6,d7


if __name__ == "__main__":
    main()
