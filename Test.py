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
import SimpleITK as sitk
import tensorflow as tf

import numpy as np
# from torchvision.models.vgg import vgg16

# def dice_coef_theoretical(y_pred, y_true):
#     """Define the dice coefficient
#         Args:
#         y_pred: Prediction
#         y_true: Ground truth Label
#         Returns:
#         Dice coefficient
#         """
#
#     y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
#
#     y_pred_f = tf.nn.sigmoid(y_pred)
#     y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)
#     y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)
#
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
#     dice = (2. * intersection) / (union + 0.00001)
#
#     if (tf.reduce_sum(y_pred) == 0) and (tf.reduce_sum(y_true) == 0):
#         dice = 1
#
#     return dice




def calDSI(binary_GT,binary_R):
    binary_GT = binary_GT.squeeze()
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s,DSI_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT.all() == 255 and binary_R.all() == 255:
                DSI_s += 1
            if binary_GT.all() == 255:
                DSI_t += 1
            if binary_R.all() == 255:
                DSI_t += 1
    DSI = 2*DSI_s/DSI_t
    # print(DSI)
    return DSI

# def computeQualityMeasures(lP, lT):
#     quality = dict()
#     labelPred = sitk.GetImageFromArray(lP, isVector=False)
#     labelTrue = sitk.GetImageFromArray(lT, isVector=False)
#     hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
#     hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
#     quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
#     quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()
#
#     dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
#     dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
#     quality["dice"] = dicecomputer.GetDiceCoefficient()
#
#     return quality

if __name__ == '__main__':
    # CUDA_CACHE_PATH = '/data/private/xxw993/'
    # focalloss_gamma_list = [0,1,2,3,4]
    # focalloss_alpha_list = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # for gamma in focalloss_gamma_list:
    #     for alpha in focalloss_alpha_list:
            device = 'cpu:0'
            batch_size = 1
            num_epochs = 50
            lr = 0.0001
            num_workers = 0

            model = UNet(2, 3).to(device)
            # criterion = nn.CrossEntropyLoss().to(device)
            # # criterion_2 = FocalLoss(gamma=gamma, alpha=alpha)
            # optimizer = optim.Adam(model.parameters(), lr=lr)
            continue_train = True

            train_data_path = './test'
            train_dataset = myDataset(train_data_path)
            # train_data_path = '/data/private/xxw993/data/cmr/traindataset/imgdata'
            # train_dataset = myDataset(train_data_path)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            # ckptpath = os.path.join('./trained_model')
            # # os.mkdir(ckptpath)
            #
            # writer = SummaryWriter(comment='UNet')
            #
            # loss_list = []
            start = 0
            if continue_train:

                start = 0
                model.load_state_dict(torch.load('./trained_model/weights_4.pth'))

            # for epoch in range(start, start+num_epochs):
            #     dt_size = len(train_loader.dataset)
            #     epoch_loss = 0
            step = 0
            for x, y in train_loader:
                    step += 1
                    x = x.to(device)
                    labels = y.to(device)

                    classfication = model(x)
                    # loss = criterion(classfication, labels.long())
                    # loss_reconstruction = criterion_2(reconstruction, x)
                    # loss = loss_classfication# + loss_reconstruction * 0.1

                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()

                    # array_1 = classfication.data.cpu().numpy()
                    output = torch.argmax(classfication, 1).squeeze()
                    array1 = output.numpy()
                    array1 = array1 * 255

                    array3 = labels.numpy()
                    array3 = array3 * 255

                    plt.figure()
                    plt.imshow(array1)
                    plt.axis('off')
                    plt.show()

                    # quality = computeQualityMeasures(array1, array3)

                    print('（1）DICE计算结果，      DSI       = {0:.4}'.format(calDSI(array1, array3)))

                    cv2.imwrite('./2d_images_output' + str(step) + '.png', array1)

            #         print("epoch:%d--%d/%d,train_loss:%0.3f" % (epoch, step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
            #         # writer.add_scalar('Loss', loss, epoch * dt_size + step)
            #         # loss_list.append([epoch * dt_size + step, loss.data.cpu().numpy])
            #     if(epoch+1)%5 == 0:
            #         torch.save(model.state_dict(), os.path.join(ckptpath, 'weights_'+str(epoch)+'.pth'))
            #         # np.save("loss.npy", np.array(loss_list))
            # torch.save(model.state_dict(), os.path.join(ckptpath, 'weights_latest.pth'))
