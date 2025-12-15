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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import combinations

#from trid_unet2 import diyUNet
from monai.networks.nets import UNet
from monai.networks.layers import Flatten
from train_func import *
import sys 
import SimpleITK as sitk
import pingouin as pg

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
    plt.colorbar()
    for i in range(len(c)):
        for j in range(len(c)):
            plt.annotate('{}\n{:.0%}'.format(c[j][i],c[j][i]/(c[j][0]+c[j][1]+c[j][2])), xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.xticks(range(3),['good','inter','poor'])
    plt.yticks(range(3), ['good', 'inter', 'poor'])
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.savefig(confu_path,bbox_inches='tight')
    #plt.show()

def train_confusion_2c(y_true, y_pred,confu_path):
    c = confusion_matrix(y_true, y_pred)
    plt.matshow(c, cmap=plt.cm.Blues)
    plt.colorbar()

    for i in range(len(c)):
        for j in range(len(c)):
            plt.annotate('{}\n{:.0%}'.format(c[j][i], c[j][i] / (c[j][0] + c[j][1])), xy=(i, j),
                         horizontalalignment='center', verticalalignment='center')
    plt.xticks(range(2), ['poor', 'non-poor'])
    plt.yticks(range(2), ['poor', 'non-poor'])
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.savefig(confu_path)
    #plt.show()


def calculate_confidence_interval(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)
    z = 1.96  # 95% 置信水平的Z值
    lower_bound = mean - (z * std / np.sqrt(n))
    upper_bound = mean + (z * std / np.sqrt(n))
    return lower_bound, mean,upper_bound

def cal_icc(cross_y_true,cross_y_pred):
    icc_dict = {
        'targets': list(range(len(cross_y_true.numpy().tolist()))) + list(range(len(cross_y_true.numpy().tolist()))),
        'raters': ['A' for i in range(len(cross_y_true.numpy().tolist()))] + ['B' for i in range(
            len(cross_y_true.numpy().tolist()))],
        'ratings': cross_y_true.numpy().tolist() + cross_y_pred.numpy().tolist()
    }
    df = pd.DataFrame(icc_dict)

    icc_results = pg.intraclass_corr(data=df, targets='targets', raters='raters', ratings='ratings')

    return icc_results


def bootstrap_auc_pairwise(y_true, preds_list, model_names=None, num_classes=3, n_bootstrap=1000, average='macro',
                           seed=42):
    np.random.seed(seed)
    n = len(y_true)
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    num_models = len(preds_list)
    if model_names is None:
        model_names = [f'Model_{i + 1}' for i in range(num_models)]

    # 保存结果
    diff_matrix = np.zeros((num_models, num_models))
    pval_matrix = np.ones((num_models, num_models))

    for (i, j) in combinations(range(num_models), 2):
        auc_diffs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(np.arange(n), size=n, replace=True)
            y_true_sample = y_true_bin[indices]
            pred_i = preds_list[i][indices]
            pred_j = preds_list[j][indices]
            try:
                auc_i = roc_auc_score(y_true_sample, pred_i, average=average, multi_class='ovr')
                auc_j = roc_auc_score(y_true_sample, pred_j, average=average, multi_class='ovr')
                auc_diffs.append(auc_i - auc_j)
            except ValueError:
                continue  # 某次 bootstrap 后类缺失，跳过

        auc_diffs = np.array(auc_diffs)
        mean_diff = np.mean(auc_diffs)
        p_value = np.mean(np.abs(auc_diffs) >= np.abs(mean_diff))

        diff_matrix[i, j] = mean_diff
        pval_matrix[i, j] = p_value

    return diff_matrix, pval_matrix, model_names






def multilayer_mip(image, layer_num=22, axis=2):
    split_layers = np.array_split(image, layer_num, axis=axis)
    mip_list = [
        np.max(layer, axis=axis)
        for layer in split_layers
    ]
    mip_image = np.stack(mip_list, axis=axis)
    return mip_image

