# demo
import numpy as np
import os
from skimage import io
from skimage.transform import resize

import glob
from measures import compute_ave_MAE_of_methods


## 0. =======set the data path=======
print("------0. set the data path------")

# >>>>>>> Follows have to be manually configured <<<<<<< #
data_name = r'DUTS'  # this will be drawn on the bottom center of the figures
data_dir = r'../../../datasets/DUTS-TE/'  # set the data directory,
                          #u ground truth and results to-be-evaluated should be in this directory
                          # the figures of PR and F-measure curves will be saved in this directory as well
gt_dir = 'DUTS-TE-Mask' # set the ground truth folder name
rs_dirs = ['DUTS-TE-Res-U2net/u2netp_epoch_1347_bce_itr_383680_train_0.25893394684588367_tar_0.022625727974809707/2021.02.09-19.40',
           'DUTS-TE-Res-U2net/u2netp_ephoch_747_bce_itr_230912_train_0.2811758668712256_tar_0.025253941847371276_valloss_0.9987935651200158/2021.02.11-17.11',
           'DUTS-TE-Res-U2net/u2netp_epoch_747_bce_itr_45760_train_0.29373008854348553_tar_0.026396556153089147/2021.02.11-16.59',
           'DUTS-TE-Res-U3net/u3netp_ephoch_636_bce_itr_85888_train_0.27360771753063257_tar_0.02309472444837659_valloss_3.708931065395347/2021.02.09-19.54',
           'DUTS-TE-Res-U3net/u3netp_ephoch_747_bce_itr_242176_train_0.2612400477783839_tar_0.022259863834981596_valloss_1.5054062861323017/2021.02.11-16.35']  # set the folder names of different methods
                        # 'rs1' contains the result of method1
                        # 'rs2' contains the result of method 2
                        # we suggest to name the folder as the method names because they will be shown in the figures' legend
lineSylClr = ['r-', 'r--', 'r*', 'g-', 'g--'] # curve style, same size with rs_dirs
linewidth = [1, 1, 1, 2, 2] # line width, same size with rs_dirs
# >>>>>>> Above have to be manually configured <<<<<<< #

gt_name_list = glob.glob(data_dir+gt_dir+'/'+'*.png') # get the ground truth file name list
# gt_name_list = os.listdir(os.path.join(data_dir, gt_dir))

## get directory list of predicted maps
rs_dir_lists = []
for i in range(len(rs_dirs)):
    rs_dir_lists.append(data_dir+rs_dirs[i]+'/')
print('\n')


## 1. =======compute the average MAE of methods=========
print("------1. Compute the average MAE of Methods------")
aveMAE, gt2rs_mae = compute_ave_MAE_of_methods(gt_name_list,rs_dir_lists)
print('\n')
for i in range(0,len(rs_dirs)):
    print('>>%s: num_rs/num_gt-> %d/%d, aveMAE-> %.3f'%(rs_dirs[i], gt2rs_mae[i], len(gt_name_list), aveMAE[i]))


## 2. =======compute the Precision, Recall and F-measure of methods=========
from measures import compute_PRE_REC_FM_of_methods,plot_save_pr_curves,plot_save_fm_curves

print('\n')
print("------2. Compute the Precision, Recall and F-measure of Methods------")
PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_name_list,rs_dir_lists,beta=0.3)
for i in range(0,FM.shape[0]):
    print(">>", rs_dirs[i],":", "num_rs/num_gt-> %d/%d,"%(int(gt2rs_fm[i][0]),len(gt_name_list)), "maxF->%.3f, "%(np.max(FM,1)[i]), "meanF->%.3f, "%(np.mean(FM,1)[i]))
print('\n')


## 3. =======Plot and save precision-recall curves=========
print("------ 3. Plot and save precision-recall curves------")
plot_save_pr_curves(PRE, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                    REC, # numpy array (num_rs_dir,255)
                    method_names = rs_dirs, # method names, shape (num_rs_dir), will be included in the figure legend
                    lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
                    linewidth = linewidth, # curve width, shape (num_rs_dir)
                    xrange = (0.5,1.0), # the showing range of x-axis
                    yrange = (0.5,1.0), # the showing range of y-axis
                    dataset_name = data_name, # dataset name will be drawn on the bottom center position
                    save_dir = data_dir, # figure save directory
                    save_fmt = 'png') # format of the to-be-saved figure
print('\n')

## 4. =======Plot and save F-measure curves=========
print("------ 4. Plot and save F-measure curves------")
plot_save_fm_curves(FM, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                    mybins = np.arange(0,256),
                    method_names = rs_dirs, # method names, shape (num_rs_dir), will be included in the figure legend
                    lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
                    linewidth = linewidth, # curve width, shape (num_rs_dir)
                    xrange = (0.0,1.0), # the showing range of x-axis
                    yrange = (0.0,1.0), # the showing range of y-axis
                    dataset_name = data_name, # dataset name will be drawn on the bottom center position
                    save_dir = data_dir, # figure save directory
                    save_fmt = 'png') # format of the to-be-saved figure
print('\n')

## 5. =======max F-measure curves=========
print("------ 5. report on maximal F-score------")
# todo: add here the maximum value of f-score

print('Done!!!')
