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
from model import U3NETP

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def calc_mae(image_name, pred):
    test_label_full_path = r'../../..//datasets/DUTS-TE/DUTS-TE-Mask'
    image_gt_mask = io.imread(os.path.join(test_label_full_path,os.path.basename(image_name)).replace('.jpg', '.png'))
    image_gt_mask = image_gt_mask/255

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im_predict = Image.fromarray(predict_np)
    imo_predict = im_predict.resize((image_gt_mask.shape[1],image_gt_mask.shape[0]),resample=Image.BILINEAR)
    pb_np_predict = np.array(imo_predict)
    if len(np.shape(image_gt_mask))==3:
        print(f'{image_name} is with 3 channels and excluded')
        return np.nan
    single_mae = np.sum(np.sum(np.abs(pb_np_predict-image_gt_mask)))/(image_gt_mask.shape[1]*image_gt_mask.shape[0])
    return single_mae

def save_output(image_name,pred,d_dir):
    test_label_full_path = r'../../..//datasets/DUTS-TE/DUTS-TE-Mask'
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np_predict = np.array(imo)

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
    # model_name= 'u3netp'

    # mode = 'save_images'
    mode = 'mae_acc'

    # RUN_ON_GPU = False
    RUN_ON_GPU = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    image_ = r'../../../datasets/DUTS-TE/DUTS-TE-Image/'
    image_dir = image_
    # image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    # prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    prediction_dir = r'../../../final_project_results/prediction/'
    # prediction_dir = r'/home/nadav/dl_seminar/final_project_results/prediction/'
    cur_date_time = time.strftime("%Y.%m.%d-%H.%M")
    prediction_dir = os.path.join(prediction_dir,model_name +'_' + cur_date_time + '_' + mode) + os.sep
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    # model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    model_dir = r'/media/nadav/final_project_results/models/2021.01.26-22.09/u2netp_epoch_1347_bce_itr_383680_train_0.25893394684588367_tar_0.022625727974809707.pth'

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

        if mode == 'save_images':
            # save results to test_results folder
            save_output(img_name_list[i_test],pred,prediction_dir)
        if mode == 'mae_acc':
            cur_image_mae = calc_mae(img_name_list[i_test], pred)
            all_images_mae.append(cur_image_mae)


        del d1,d2,d3,d4,d5,d6,d7

    if mode == 'mae_acc':
        average_mae = np.nanmean(all_images_mae)
        mae_dict = {
            'average_mae': average_mae,
            'mae_vec': all_images_mae,
            'images_list':img_name_list
        }
        print(average_mae)
        json.dump(mae_dict, open(prediction_dir + 'average_mae.json', 'w'))

if __name__ == "__main__":
    main()
