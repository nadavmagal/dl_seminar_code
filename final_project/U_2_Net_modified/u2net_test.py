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

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):
    test_label_full_path = r'../../..//datasets/DUTS-TE/DUTS-TE-Mask'
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

    test_mask = io.imread(os.path.join(test_label_full_path,os.path.basename(image_name)).replace('.jpg', '.png'))
    # conc_im = np.hstack((image, imo, test_mask))

    # imo.save(d_dir+imidx+'.png')
    # conc_im.save(d_dir+imidx+'.png')

    plt.figure(figsize=[15, 10], dpi=500)
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title('image')
    plt.subplot(1,3,2)
    plt.imshow(imo)
    plt.title('prediction')
    plt.subplot(1,3,3)
    plt.imshow(test_mask)
    plt.title('mask')
    plt.savefig(d_dir+imidx+'.png', dpi=300)
    plt.close('all')


def main():

    # --------- 1. get image path and name ---------
    # model_name='u2net'#u2netp
    model_name= 'u2netp'

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    image_ = r'../../../datasets/DUTS-TE/DUTS-TE-Image/'
    image_dir = image_
    # image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    # prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    prediction_dir = r'../../../final_project_results/prediction/'
    # prediction_dir = r'/home/nadav/dl_seminar/final_project_results/prediction/'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")
    prediction_dir = os.path.join(prediction_dir, cur_date_time) + os.sep
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    # model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    # model_dir = r'/media/nadav/final_project_results/models/2021.01.19-19.49/u2net_bce_itr_94000_train_0.300097_tar_0.028924.pth'
    # model_dir = r'../../../final_project_results/models/2021.01.22-12.01/u2netp_ephoch_318_bce_itr_280720_train_0.3595311115919189_tar_0.033946142557331103.pth'
    model_dir = r'/media/nadav/final_project_results/models/2021.01.22-12.01/u2netp_ephoch_504_bce_itr_444400_train_0.3133861400017684_tar_0.02861546263818375.pth'
    params_to_save = dict()
    params_to_save['model_path'] = model_dir
    json.dump(params_to_save, open(prediction_dir+'models_params.json','w'))

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

    # net.load_state_dict(torch.load(c))
    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
