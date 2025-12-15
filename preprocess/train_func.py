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
from monai.transforms import Compose,LoadImaged,EnsureChannelFirstd,RandRotated,RandAffined,SaveImaged
import random
from sklearn.metrics import confusion_matrix
import codecs

#from trid_unet2 import diyUNet
from monai.networks.nets import UNet
from monai.networks.layers import Flatten
from train_func import *
import sys 
import SimpleITK as sitk

def f1_score_b(y_true, y_pred):
    """
    计算F1-score
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: F1-score
    """
    #y_true = torch.Tensor(y_true)
    #y_pred = torch.Tensor(y_pred)
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1 - y_true) * y_pred)
    fn = torch.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1.item()


class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        
        pt = _input
        alpha = self.alpha
        eps=1e-7
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt+eps) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt+eps)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, num_class=2, alpha=0.6, gamma=2, balance_index=0, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
        gamma = self.gamma
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Focal_Loss():

# ***二分类Focal Loss***

	def __init__(self,alpha=0.25,gamma=2):
		super(Focal_Loss,self).__init__()
		self.alpha=alpha
		self.gamma=gamma
	
	def forward(self,preds,labels):
		
		#preds:sigmoid的输出结果
		#labels：标签

		eps=1e-7
		loss_1=-1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
		loss_0=-1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
		loss=loss_0+loss_1
		return torch.mean(loss)

# ******创建t_scores,去除无score的图像*******
def cre_scores(excel_path,head_path):
    ecl1 = pd.read_excel(excel_path, sheet_name='Sheet1', keep_default_na=False)
    nrows = ecl1.shape[0]
    ncols = ecl1.shape[1]
    t_scores = []

    for i in range(nrows):
        nstr = ecl1.iloc[i, 0]
        dename = "{}/PRoveIT-{}".format(head_path, nstr[-6:])
        if os.path.exists(dename):
            if ecl1.iloc[i, ncols-1] == 1 or ecl1.iloc[i, ncols-1] == 2 or ecl1.iloc[i, ncols-1] == 3:
                t_scores.append(ecl1.iloc[i, ncols-1]-1)
            else:shutil.rmtree(dename)

    return t_scores


def train_confusion(y_true, y_pred,confu_path):
    c = confusion_matrix(y_true, y_pred)
    plt.matshow(c, cmap=plt.cm.Reds)
    for i in range(len(c)):
        for j in range(len(c)):
            plt.annotate(c[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(confu_path)
    #plt.show()

def cre_dict(head_path,name_scores):
    imglist = sorted(glob(os.path.join(head_path, '*', 'mCTA2.nii.gz')))
    lab1 = sorted(glob(os.path.join(head_path, '*', 'rbrain.nii.gz')))
    lab2 = sorted(glob(os.path.join(head_path, '*', 'Vascular_territories_2sides.nii.gz')))
    t_scores = []
    for img in imglist:
        fname,lname = os.path.split(img)
        mname = os.path.split(fname)[-1]
        mname = mname[:14]
        t_scores.append(name_scores[mname])
    data_dict = [{'image': image, 'lab1': lab1, 'lab2': lab2, 'label': label} for image, lab1, lab2, label in zip(imglist, lab1, lab2, t_scores)]
    return data_dict

def cre_dict_extra(extra_path):
    imglist_extra = sorted(glob(os.path.join(extra_path, '*', '*','mCTA2_trans.nii.gz')))
    lab1_extra = sorted(glob(os.path.join(extra_path, '*', '*','rbrain_trans.nii.gz')))
    lab2_extra = sorted(glob(os.path.join(extra_path, '*', '*','Vascular_territories_2sides_trans.nii.gz')))
    t_scores_extra =np.ones(len(imglist_extra),dtype=np.int8)
    data_dict_extra = [{'image': image, 'lab1': lab1, 'lab2': lab2, 'label': label} for image, lab1, lab2, label in zip(imglist_extra, lab1_extra, lab2_extra, t_scores_extra)]
    return data_dict_extra

def cre_extra(train_d1,extra_path):
    if(os.path.exists(extra_path)):
        shutil.rmtree(extra_path)
        os.mkdir(extra_path)
    else:os.mkdir(extra_path)
    pre_transform = Compose(
        [
        LoadImaged(keys=['image', 'lab1', 'lab2']),
        EnsureChannelFirstd(keys=['image', 'lab1', 'lab2']),
        #RandRotated(keys=['image', 'lab1', 'lab2'], range_x=0.3,prob=1),
        ]
    )
    for img in train_d1:
        dir,fname=os.path.splitext(img['image'])
        name=dir.split('/')[-2]
        #data_rotate=pre_transform(img)
        loader = LoadImaged(keys=['image', 'lab1', 'lab2'])
        img = loader(img)
        #add_channel_first = EnsureChannelFirstd(keys=['image', 'lab1', 'lab2'])
        #data_dict_addc = add_channel_first(img)
        rotate = RandRotated(keys=['image', 'lab1', 'lab2'], range_x=0.3,prob=1)
        img = rotate(img)
        translation = RandAffined(keys=['image', 'lab1', 'lab2'],prob=0.5,translate_range=25)
        dataout = translation(img)
        #flip = RandFlipd(keys=['image', 'lab1','lab2'],prob=1,spatial_axis=0)
        save = SaveImaged(keys=['image', 'lab1', 'lab2'],output_dir=extra_path+'/'+name+'_r',output_ext='.nii.gz',output_dtype=np.int16)
        save(dataout)

def minmax_dim(x,y,z,img_array):
        xleft=[]
        flag = 0
        for k in range(z):
            for i in range(x):
                for j in range(y):
                    if img_array[i,j,k] !=0:
                        xleft.append(i)
                        flag = 1
                        break
                if flag ==1:
                    flag = 0
                    break
        x_l_last = min(xleft)

        xright=[]
        flag = 0
        for k in range(z):
            for i in range(x):
                for j in range(y):
                    if img_array[x-i-1,j,k] !=0:
                        xright.append(x-i-1)
                        flag = 1
                        break
                if flag ==1:
                    flag = 0
                    break
        x_r_last = max(xright)
        return x_l_last,x_r_last

class pre_process:
    def __init__(self,train_path='',test_path='',extra_path=''):
        self.train_path = train_path
        self.test_path = test_path
        self.extra_path = extra_path
    def pre_dict(img_path):

        return

    def pre_load(self,img_dict,key):
        img = sitk.ReadImage(img_dict[key])
        img_array = sitk.GetArrayFromImage(img).transpose(2,1,0)
        img_size = img_array.shape
        hdr = [img.GetDirection(),img.GetOrigin(),img.GetSpacing()]
        return img_array,img_size,hdr
    
    def pre_crop(self,img_array,img_size):
        #data_rotate=pre_transform(img)
        #loader = LoadImaged(keys=['image', 'lab1', 'lab2'])
        #img = loader(img)
        
        xl,xr = minmax_dim(img_size[0],img_size[1],img_size[2],img_array)
        yl,yr = minmax_dim(img_size[1],img_size[0],img_size[2],img_array.transpose(1,0,2))
        zl,zr = minmax_dim(img_size[2],img_size[1],img_size[0],img_array.transpose(2,1,0))
        return [xl,xr,yl,yr,zl,zr]
        
    def pre_save(self,img_array,hdr,save_path):
        out = sitk.GetImageFromArray(img_array.transpose(2,1,0))
        out.SetDirection(hdr[0])
        out.SetOrigin(hdr[1])
        out.SetSpacing(hdr[2])
        sitk.WriteImage(out,save_path)
   
    def pre_flip(self,img_array,axis):
        img_array = np.flip(img_array,axis)
        return img_array


def copyfile(data_d2):
    path1 = '/home/liuhao/train/test/new/train'
    path2 = '/home/liuhao/train/test/new/test'
    for data in data_d2[0:27]:
        fpath,lpath = os.path.split(data['image'])
        mpath = os.path.split(fpath)[-1]
        shutil.copytree(fpath,path1+'/'+mpath)

    for data in data_d2[27:39]:
        fpath,lpath = os.path.split(data['image'])
        mpath = os.path.split(fpath)[-1]
        shutil.copytree(fpath,path2+'/'+mpath)

def crop_all(data_d1):
    
    pre_process = pre_process()
    for data in data_d1[0:2]:
        fpath,lpath = os.path.split(data['image'])
        fpath1,lpath1 = os.path.split(data['lab1'])
        fpath2,lpath2 = os.path.split(data['lab2'])
        mpath = os.path.split(fpath)[-1]

        img_array,img_size,hdr = pre_process.pre_load(data,'image')
        img_array1,img_size1,hdr1 = pre_process.pre_load(data,'lab1')
        img_array2,img_size2,hdr2 = pre_process.pre_load(data,'lab2')
        para = pre_process.pre_crop(img_array,img_size)
        img_array = img_array[para[0]:para[1]+1,para[2]:para[3]+1,para[4]:para[5]+1]
        img_array1 = img_array1[para[0]:para[1]+1,para[2]:para[3]+1,para[4]:para[5]+1]
        img_array2 = img_array2[para[0]:para[1]+1,para[2]:para[3]+1,para[4]:para[5]+1]
        pre_process.pre_save(img_array,hdr,'/home/liuhao/train/test/new/extra/d1/'+mpath+'/'+lpath)
        pre_process.pre_save(img_array1,hdr1,'/home/liuhao/train/test/new/extra/d1/'+mpath+'/'+lpath1)
        pre_process.pre_save(img_array2,hdr2,'/home/liuhao/train/test/new/extra/d1/'+mpath+'/'+lpath2)
 
def flip_d1():
    pre_process = pre_process()
    for data in data_d1:
        fpath,lpath = os.path.split(data['image'])
        fpath1,lpath1 = os.path.split(data['lab1'])
        fpath2,lpath2 = os.path.split(data['lab2'])
        mpath = os.path.split(fpath)[-1]

        img_array,img_size,hdr = pre_process.pre_load(data,'image')
        img_array1,img_size1,hdr1 = pre_process.pre_load(data,'lab1')
        img_array2,img_size2,hdr2 = pre_process.pre_load(data,'lab2')
        print(type(img_array[0][0][0]))
        img_array=pre_process.pre_flip(img_array,0)
        img_array1=pre_process.pre_flip(img_array1,0)
        img_array2=pre_process.pre_flip(img_array2,0)
        os.mkdir('/home/liuhao/train/test/new/extra/d1/'+mpath)
        pre_process.pre_save(img_array,hdr,'/home/liuhao/train/test/new/extra/d1/'+mpath+'/'+lpath)
        pre_process.pre_save(img_array1,hdr1,'/home/liuhao/train/test/new/extra/d1/'+mpath+'/'+lpath1)
        pre_process.pre_save(img_array2,hdr2,'/home/liuhao/train/test/new/extra/d1/'+mpath+'/'+lpath2)

def pre_mip():
    dataset_3D_path = '/home/liuhao/train/test/2D'
    dataset_2D_path = '/home/liuhao/train/test/mass'

    for lab in os.listdir(dataset_3D_path):
        ori_dir = os.path.join(dataset_3D_path, lab)
        dest_dir = os.path.join(dataset_2D_path, lab)

        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        for name in os.listdir(ori_dir):
            if name == 'NCCT_ATLAS_MNI_brain_mask.nii.gz':
                continue
            else:
                ni_image_path = os.path.join(ori_dir, name)
                MIP_image_path = os.path.join(dest_dir, name)

                ni_image = nib.load(ni_image_path).get_fdata()
                
                ni_image = np.max(ni_image, axis=2)

                thresh = np.percentile(ni_image, 99.9)
                MIP_image = np.clip(ni_image, 0, thresh)

                #monai.data.write_nifti(MIP_image, MIP_image_path)
                writer = monai.data.NibabelWriter(output_dtype=np.uint16)
                writer.set_data_array(MIP_image, channel_dim=None)
                #writer.set_metadata({"spatial_shape": (5, 5)})
                writer.write(MIP_image_path,verbose=True)
                print('Process done!')

def multilayer_mip(image, layer_num=22, axis=2):
    split_layers = np.array_split(image, layer_num, axis=axis)
    mip_list = [
        np.max(layer, axis=axis)
        for layer in split_layers
    ]
    mip_image = np.stack(mip_list, axis=axis)
    return mip_image

