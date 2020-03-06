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

import numpy as np
# from torchvision.models.vgg import vgg16

if __name__ == '__main__':
    # CUDA_CACHE_PATH = '/data/private/xxw993/'
    # focalloss_gamma_list = [0,1,2,3,4]
    # focalloss_alpha_list = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # for gamma in focalloss_gamma_list:
    #     for alpha in focalloss_alpha_list:
            device = 'cpu:0'
            batch_size = 10
            num_epochs = 50
            lr = 0.0001
            num_workers = 0

            model = UNet(2, 3).to(device)
            criterion = nn.CrossEntropyLoss().to(device)
            # criterion_2 = FocalLoss(gamma=gamma, alpha=alpha)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            continue_train = True

            train_data_path = './'
            train_dataset = myDataset(train_data_path)
            # train_data_path = '/data/private/xxw993/data/cmr/traindataset/imgdata'
            # train_dataset = myDataset(train_data_path)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            ckptpath = os.path.join('./trained_model')
            # os.mkdir(ckptpath)

            writer = SummaryWriter(comment='UNet')

            loss_list = []
            start = 0
            if continue_train:

                start = 0
                model.load_state_dict(torch.load('./trained_model/model1.pth'))

            for epoch in range(start, start+num_epochs):
                dt_size = len(train_loader.dataset)
                epoch_loss = 0
                step = 0
                for x, y in train_loader:
                    step += 1
                    x = x.to(device)
                    labels = y.to(device)

                    classfication = model(x)
                    loss = criterion(classfication, labels.long())
                    # loss_reconstruction = criterion_2(reconstruction, x)
                    # loss = loss_classfication# + loss_reconstruction * 0.1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # output = torch.argmax(classfication, 1).squeeze()
                    # array1 = output.numpy()
                    # array2 = array1[1]
                    # array2 = array2 * 255
                    #
                    # plt.figure()
                    # plt.imshow(array2)
                    # plt.axis('off')
                    # plt.show()
                    # array_1 = classfication.data.cpu().numpy()


                    print("epoch:%d--%d/%d,train_loss:%0.3f" % (epoch, step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
                    # writer.add_scalar('Loss', loss, epoch * dt_size + step)
                    # loss_list.append([epoch * dt_size + step, loss.data.cpu().numpy])
                if(epoch+1)%3 == 0:
                    torch.save(model.state_dict(), os.path.join(ckptpath, 'weights_'+str(epoch)+'.pth'))
                    # np.save("loss.npy", np.array(loss_list))
            torch.save(model.state_dict(), os.path.join(ckptpath, 'weights_latest.pth'))
