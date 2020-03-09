from torch.utils.data import Dataset
import os
import cv2
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# from dataset.cmrdataset import normalize_cmr
from libtiff import TIFF
from PIL import Image
import tifffile as tif
import torch

def make_dataset(root):
    imgs = []
    n = len(os.listdir(root)) // 2
    # for i in range(n):
    #     img = os.path.join(root, "%03d.png" % i)
    #     mask = os.path.join(root, "%03d_mask.png" % i)
    #     imgs.append(s(img, mask))

    for f in os.listdir(os.path.join(root, '2d_images')):
            if 'tif' not in f:
                continue
            img = os.path.join(root, '2d_images', f)
            mask = os.path.join(root, '2d_masks', f)
            imgs.append((img, mask))

    return imgs

class myDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = cv2.imread(x_path)
        img_y = cv2.imread(y_path)
        # print(x_path, y_path)
        # print(img_x.shape, img_y.shape)
        # print(x_path)
        # img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
        img_y = cv2.cvtColor(img_y, cv2.COLOR_BGR2GRAY)
        img_y[img_y>0] = 1

        # img_y = normalize_cmr(img_y)
        # img_x = Image.open(x_path)
        # img_y = Image.open(y_path)
        img_x = transforms.ToTensor()(img_x)
        img_y = torch.from_numpy(img_y)
        #img_y = transforms.ToTensor()(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

# if __name__ == '__main__':
#     train_data_path = './'
#     train_dataset = myDataset(train_data_path)
#     # train_data_path = '/data/private/xxw993/data/cmr/traindataset/imgdata'
#     # train_dataset = myDataset(train_data_path)
#     train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
#
#     for epoch in range(1):
#         dt_size = len(train_loader.dataset)
#         epoch_loss = 0
#         step = 0
#         for x, y in train_loader:
#             print(x.shape, y.shape)