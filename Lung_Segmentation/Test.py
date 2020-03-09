import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import os
from dataset import myDataset
# from dataset.datasetMRI import CPM17Dataset
from tensorboardX import SummaryWriter
# from net.UNet_N import UNet
# from net.SMNet import SMNet
# from Loss.FocalLoss import FocalLoss
from models.UNet_N import UNet
from matplotlib import pyplot as plt
import cv2
import csv
import SimpleITK as sitk
import tensorflow as tf

import numpy as np


# 计算DICE系数，即DSI
# def calDSI(binary_GT, binary_R):
#     row, col = binary_GT.shape  # 矩阵的行与列
#     DSI_s, DSI_t = 0, 0
#     for i in range(row):
#         for j in range(col):
#             if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
#                 DSI_s += 1
#             if binary_GT[i][j] == 255:
#                 DSI_t += 1
#             if binary_R[i][j] == 255:
#                 DSI_t += 1
#     DSI = 2 * DSI_s / DSI_t
#     # print(DSI)
#     return DSI


# # 计算VOE系数，即VOE
# def calVOE(binary_GT, binary_R):
#     row, col = binary_GT.shape  # 矩阵的行与列
#     VOE_s, VOE_t = 0, 0
#     for i in range(row):
#         for j in range(col):
#             if binary_GT[i][j] == 255:
#                 VOE_s += 1
#             if binary_R[i][j] == 255:
#                 VOE_t += 1
#     VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
#     return VOE


# # 计算RVD系数，即RVD
# def calRVD(binary_GT, binary_R):
#     row, col = binary_GT.shape  # 矩阵的行与列
#     RVD_s, RVD_t = 0, 0
#     for i in range(row):
#         for j in range(col):
#             if binary_GT[i][j] == 255:
#                 RVD_s += 1
#             if binary_R[i][j] == 255:
#                 RVD_t += 1
#     RVD = RVD_t / RVD_s - 1
#     return RVD
#
#
# # 计算Prevision系数，即Precison
# def calPrecision(binary_GT, binary_R):
#     row, col = binary_GT.shape  # 矩阵的行与列
#     P_s, P_t = 0, 0
#     for i in range(row):
#         for j in range(col):
#             if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
#                 P_s += 1
#             if binary_R[i][j] == 255:
#                 P_t += 1
#
#     Precision = P_s / P_t
#     return Precision
#
#
# # 计算Recall系数，即Recall
# def calRecall(binary_GT, binary_R):
#     row, col = binary_GT.shape  # 矩阵的行与列
#     R_s, R_t = 0, 0
#     for i in range(row):
#         for j in range(col):
#             if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
#                 R_s += 1
#             if binary_GT[i][j] == 255:
#                 R_t += 1
#
#     Recall = R_s / R_t
#     return Recall

def caculate_dice(target, gt):
    tp = target * gt
    dice = (2 * tp.sum()) / (target.sum() + gt.sum() + 1e-20)
    return dice

def caculate_f1(target, gt):
    h, w = target.shape[:2]
    area = h * w
    precision = (target==gt).sum() / area

    b = np.array(target)
    b[b==0] = 100
    c = np.array(gt)
    c[c==0] = 101
    d = b - c
    recall = (b==c).sum()/(gt[gt!=0].sum()+1)
    f1 = (2 * recall * precision) / (recall + precision)
    return f1
def caculate_F1(target, gt):
    h, w = target.shape[:2]
    area = h * w
    truth = 1 - np.bitwise_xor(target.astype(np.uint8), gt.astype(np.uint8))
    precision = truth.sum() / area

    tp = np.bitwise_and(target.astype(np.uint8), gt.astype(np.uint8)).sum()
    recall = tp / gt.sum()
    f1 = (2 * recall * precision) / (recall + precision)
    return f1


if __name__ == '__main__':

            device = 'cpu:0'
            batch_size = 1
            num_epochs = 50
            lr = 0.0001
            num_workers = 0

            model = UNet(2, 3).to(device)

            continue_train = True

            train_data_path = './test'
            train_dataset = myDataset(train_data_path)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


            start = 0
            if continue_train:

                start = 0
                model.load_state_dict(torch.load('./trained_model/model1.pth'))

            model.eval()
            dice_sum = 0
            f1_sum = 0


            step = 0
            for x, y in train_loader:
                    step += 1
                    x = x.to(device)
                    labels = y.to(device)

                    classfication = model(x)

                    output = torch.argmax(classfication, 1).squeeze()
                    array1 = output.numpy()

                    array3 = labels.numpy()

                    dice = caculate_dice(array1, array3)
                    dice_sum += dice

                    plt.figure()
                    plt.imshow(array1)
                    plt.axis('off')
                    plt.show()

                    inputs = x.squeeze().data.cpu().numpy() * 255
                    array1 = array1 * 255
                    array3 = array3 * 255
                    array3 = array3.squeeze()
                    inputs = inputs[1]
                    img_merge = np.concatenate([inputs, array1, array3], axis=1)

                    print(dice)

                    cv2.imwrite('./test/concatenate_image/' + str(step) + '_.png', img_merge)

                    if step > 80:
                        break

                    #

                    # file_path = "./test/2d_masks"
                    # ds = gdal.Open(file_path)
                    # driver = gdal.GetDriverByName('PNG')
                    # dst_ds = driver.CreateCopy('./test/example'+str(step)+'.png', ds)
                    # dst_ds = None
                    # src_ds = None

                    # # step 1：读入图像，并灰度化
                    # img_GT = cv2.imread('./test/example'+str(step)+'.png', 0)
                    # img_R = cv2.imread('./2d_images_output' + str(step) + '.png', 0)


                    # step 2: 二值化
                    # ret_GT, binary_GT = cv2.threshold(array1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    # ret_R, binary_R = cv2.threshold(array3, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                    # step 4：计算DSI
            print(dice_sum / (step))

                    # # step 5：计算VOE
                    # print('（2）VOE计算结果，       VOE       = {0:.4}'.format(calVOE(binary_GT, binary_R)))

                    # # step 6：计算RVD
                    # print('（3）RVD计算结果，       RVD       = {0:.4}'.format(calRVD(binary_GT, binary_R)))
                    #
                    # # step 7：计算Precision
                    # print('（4）Precision计算结果， Precision = {0:.4}'.format(calPrecision(binary_GT, binary_R)))
                    #
                    # # step 8：计算Recall
                    # print('（5）Recall计算结果，    Recall    = {0:.4}'.format(calRecall(binary_GT, binary_R)))

                    # cv2.imwrite('./2d_images_output' + str(step) + '.png', array1)


