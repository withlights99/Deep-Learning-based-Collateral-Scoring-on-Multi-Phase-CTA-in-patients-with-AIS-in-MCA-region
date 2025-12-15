import shutil
import numpy as np
import pandas as pd
import nibabel as nb
import os
import copy
import torch
import torch.nn as nn
import time
from torch.optim import Adam
import matplotlib.pyplot as plt
from glob import glob
import monai
from monai.data import Dataset
import random
import codecs
from monai.transforms import Compose,LoadImaged,EnsureChannelFirstd,ToTensord,Resized,ScaleIntensityRanged,ConcatItemsd,RandFlipd,ScaleIntensityd,SqueezeDimd
#from trid_unet2 import diyUNet
from monai.networks.nets import UNet,DenseNet121,resnet10,resnet18



import initial


#sys.path.insert(0, '../build/SimpleITK-build/Wrapping/Python/SimpleITK.py')
#import SimpleITK as sitk

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

# ******transforms*******
train_transform = Compose(
    [
        LoadImaged(keys=['image1','image2','image3']),
        # ScaleIntensityRanged(keys=['image1','image2','image3'], a_min=0, a_max=350, b_min=0.0, b_max=1.0, clip=True, dtype=np.float32),
        #ScaleIntensityd(keys='image',minv=0.0, maxv=1.0),

        EnsureChannelFirstd(keys=['image1','image2','image3']),
        # SqueezeDimd(keys=['image1','image2','image3'],dim=2),
        #Resized(keys=['image'], spatial_size=(spa_size, spa_size, spa_size), size_mode='all'),
        # RandFlipd(keys=['image1','image2','image3'],spatial_axis=2 ,prob=0.5),
        # RandRotated(keys=['image1','image2','image3'], range_x=0.1,prob=0.5),
        #RandAffined(keys=['image', 'lab1', 'lab2'],prob=0.5,translate_range=5),
        ConcatItemsd(keys=['image1','image2','image3'], name='inputs'),
        ToTensord(keys='inputs')
    ]
)
transform = Compose(
    [
        LoadImaged(keys=['image1', 'image2', 'image3']),
        # ScaleIntensityRanged(keys=['image1', 'image2', 'image3'], a_min=0, a_max=350, b_min=0.0, b_max=1.0, clip=True,dtype=np.float32),
        #ScaleIntensityRanged(keys=['image'], a_min=-20, a_max=350, b_min=0.0, b_max=1.0, clip=True, dtype=np.float32),
        #SqueezeDimd(keys='image',dim=2),
        #ScaleIntensityRanged(keys=['image'], a_min=-20, a_max=400, b_min=0.0, b_max=1.0, clip=True, dtype=np.float32),

        EnsureChannelFirstd(keys=['image1','image2','image3']),
        # SqueezeDimd(keys=['image1', 'image2', 'image3'],dim=2),

        # Resized(keys=['image'], spatial_size=(spa_size, spa_size, spa_size), size_mode='all'),
        ConcatItemsd(keys=['image1','image2','image3'], name='inputs'),
        ToTensord(keys='inputs')
    ]
)



if __name__ == '__main__':

    # *********** excel读取，创建scores列表 **************
    excel_path = '/home/liuhao/train/code2/train_main/CandLVO_scores_new.xlsx'
    name_scores = initial.cre_names_scores(excel_path)



    # 数据加载******字典创建
    data_path = '/mnt/nvme1/trash'
    data_dict = initial.crea_dict(data_path, name_scores)



    print(data_dict)

    test_ds = Dataset(data=data_dict, transform=transform)

    test_loader = monai.data.DataLoader(
        test_ds, batch_size=1, num_workers=0
    )

    my_model = torch.load('1.pth')

    nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0,20.0]).to(device))

    with torch.no_grad():

        for batch_data in test_loader:
            b_x, b_y1 = batch_data["inputs"].to(device), batch_data["c_label"].to(device),
            my_model.eval()
            output = my_model(b_x)
            # 预测分类
            pre_lab = torch.argmax(output,dim=1)

            print('预测值是{}:1 good 2 inter 3 poor'.format(pre_lab+1))
            print('真实值是{}:1 good 2 inter 3 poor'.format(b_y1+1))











