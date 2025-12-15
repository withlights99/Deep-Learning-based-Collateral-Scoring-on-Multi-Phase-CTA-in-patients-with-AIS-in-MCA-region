import shutil
import numpy as np
import pandas as pd
import nibabel as nb
import os
import copy
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from glob import glob
import monai
import random
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from sklearn.metrics import f1_score as f1s
import codecs

#from trid_unet2 import diyUNet
from train_func import *
import sys
from roc_auc import ROC_paint,ROC_2
#sys.path.insert(0, '../build/SimpleITK-build/Wrapping/Python/SimpleITK.py')
#import SimpleITK as sitk

def cre_names_scores(excel_path):
    ecl1 = pd.read_excel(excel_path, sheet_name='Sheet1', keep_default_na=False)
    nrows = ecl1.shape[0]
    ncols = ecl1.shape[1]
    name_scores = {}

    for i in range(nrows):
        nstr = ecl1.iloc[i, 0]
        dename = "PRoveIT-{}".format(nstr[-6:])
        if ecl1.iloc[i, ncols-1] == 1 or ecl1.iloc[i, ncols-1] == 2 or ecl1.iloc[i, ncols-1] == 3:
            name_scores.update({dename:[ecl1.iloc[i,ncols-1]-1, ecl1.iloc[i,ncols-1]-1]})

    return name_scores


def cre_esnames_scores(excel_path):
    ecl1 = pd.read_excel(excel_path, sheet_name='Sheet1', keep_default_na=False)
    nrows = ecl1.shape[0]
    ncols = ecl1.shape[1]
    name_scores = {}

    for i in range(nrows):
        dename = ecl1.iloc[i, 1]

        if ecl1.iloc[i, ncols-1] == 1 or ecl1.iloc[i, ncols-1] == 2 or ecl1.iloc[i, ncols-1] == 3:
            name_scores.update({dename:[ecl1.iloc[i,ncols-1]-1, ecl1.iloc[i,ncols-1]-1]})

    return name_scores

def cre_names_scores_2c(excel_path):
    ecl1 = pd.read_excel(excel_path, sheet_name='Sheet1', keep_default_na=False)
    nrows = ecl1.shape[0]
    ncols = ecl1.shape[1]
    name_scores = {}

    for i in range(nrows):
        nstr = ecl1.iloc[i, 0]
        dename = "PRoveIT-{}".format(nstr[-6:])
        if ecl1.iloc[i, ncols-1] == 1 or ecl1.iloc[i, ncols-1] == 2:
            name_scores.update({dename: 0})
        elif ecl1.iloc[i, ncols-1] == 3:
            name_scores.update({dename: 1})

    return name_scores

def crea_dict(data_path,name_scores):
    dict = sorted(glob(os.path.join(data_path, '*')))
    dict1 = []
    dict2 = []
    dict3 = []
    # dict4 = sorted(glob(os.path.join(data_path, '*','NCCT_baseline.nii.gz')))
    name_dict = []
    c_score = []
    lvo_score = []
    for data in dict:
        fname,lname = os.path.split(data)

        if lname in name_scores:
            c_score.append(name_scores[lname][0])
            lvo_score.append(name_scores[lname][1])
            name_dict.append(lname[-6:])
            dict1.append(os.path.join(data, 'mCTA1.nii.gz'))
            dict2.append(os.path.join(data, 'mCTA2.nii.gz'))
            dict3.append(os.path.join(data, 'mCTA3.nii.gz'))

    datadict = [{'image1': image1,'image2': image2,'image3': image3,'name':name,'c_label': c_label,'lvo_label': lvo_label} for image1,image2,image3,name,c_label,lvo_label in zip(dict1,dict2,dict3,name_dict,c_score,lvo_score)]
    return datadict


def crea_esdict(data_path,name_scores):
    dict = sorted(glob(os.path.join(data_path,'*')))
    dict1 = []
    dict2 = []
    dict3 = []
    # dict4 = sorted(glob(os.path.join(data_path, '*','NCCT_baseline.nii.gz')))
    name_dict = []
    c_score = []
    lvo_score = []
    for data in dict:
        fname,lname = os.path.split(data)
        iname = int(lname[-6:-4] + lname[-3:])

        if iname in name_scores:
            c_score.append(name_scores[iname][0])
            lvo_score.append(name_scores[iname][1])
            name_dict.append('es_'+lname[-6:])
            dict1.append(os.path.join(data,'mCTA1.nii.gz'))
            dict2.append(os.path.join(data, 'mCTA2.nii.gz'))
            dict3.append(os.path.join(data, 'mCTA3.nii.gz'))

    datadict = [{'image1': image1,'image2': image2,'image3': image3,'name':name,'c_label': c_label,'lvo_label': lvo_label} for image1,image2,image3,name,c_label,lvo_label in zip(dict1,dict2,dict3,name_dict,c_score,lvo_score)]
    return datadict


def specificity(predict, target):  # Specificity，true negative rate一样
    if torch.is_tensor(predict):
        predict = predict.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(bool))
    target = np.atleast_1d(target.astype(bool))

    tn = np.count_nonzero(~predict & ~target)
    fp = np.count_nonzero(predict & ~target)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity


class variants_init():
    def __init__(self,flag):
        self.flag = flag
        if self.flag == 'global':
            self.best_train_acc1 = 0.0
            self.best_train_acc2 = 0.0
            self.least_train_loss = 10.0

            self.best_test_acc1 = 0.0
            self.best_test_acc2 = 0.0
            self.best_test_y1_true = []
            self.best_test_y1_pred = []
            #
            # self.best_f1_c_micro = 0.0
            # self.best_f1_c_macro = 0.0
            # self.best_f1_lvo_micro = 0.0
            # self.best_f1_lvo_macro = 0.0

            self.train_loss_all = []
            self.train_loss1_all = []
            self.train_loss2_all = []

            self.train_acc1_all = []
            self.train_acc2_all = []

            self.train_f1_c_all = []
            self.train_f1_c_micro_all = []
            self.train_f1_c_macro_all = []
            self.train_f1_lvo_micro_all = []
            self.train_f1_lvo_macro_all = []

            self.train_precision_c_macro_all = []
            self.train_precision_c_micro_all = []

            self.train_recall_c_macro_all = []
            self.train_recall_c_micro_all = []

            self.train_specificity_c_all = []

        #******************** test ****************
            self.test_loss_all = []
            self.test_acc1_all = []
            self.test_acc2_all = []

            self.test_f1_c_all = []
            self.test_f1_c_micro_all = []
            self.test_f1_c_macro_all = []
            self.test_f1_lvo_micro_all = []
            self.test_f1_lvo_macro_all = []

            self.test_precision_c_macro_all = []
            self.test_precision_c_micro_all = []

            self.test_recall_c_macro_all = []
            self.test_recall_c_micro_all = []

            self.test_specificity_c_all = []

            self.y1_true = torch.tensor([], dtype=torch.int)
            self.y1_pred = torch.tensor([], dtype=torch.int)
            self.y2_true = torch.tensor([], dtype=torch.int)
            self.y2_pred = torch.tensor([], dtype=torch.int)
            self.y1_score = torch.tensor([], dtype=torch.int)
            self.y2_score = torch.tensor([], dtype=torch.int)
        else:
            self.loss_sum = 0.0
            self.loss1_sum = 0.0
            self.loss2_sum = 0.0
            self.data_num = 0
            self.y1_true = torch.tensor([], dtype=torch.int)
            self.y1_pred = torch.tensor([], dtype=torch.int)
            self.y2_true = torch.tensor([], dtype=torch.int)
            self.y2_pred = torch.tensor([], dtype=torch.int)
            self.y1_score = torch.tensor([], dtype=torch.int)
            self.y2_score = torch.tensor([], dtype=torch.int)
            self.corrects1 = 0
            self.corrects2 = 0

            self.loss = 0.0
            self.loss1 = 0.0
            self.loss2 = 0.0
            self.acc1 = 0.0
            self.acc2 = 0.0

            self.f1_c = 0.0
            self.f1_micro_c = 0.0
            self.f1_macro_c = 0.0
            self.f1_micro_lvo = 0.0
            self.f1_macro_lvo = 0.0

            self.precision_macro_c = 0.0
            self.precision_micro_c = 0.0

            self.recall_macro_c = 0.0
            self.recall_micro_c = 0.0

            self.specificity_c = 0.0



    def var_updata(self,loss,output1,output2,b_x_size,b_y1,b_y2,loss1,loss2):
        sof = nn.Softmax(dim=1)
        self.loss_sum += loss.item() * b_x_size
        self.loss1_sum += loss1.item() * b_x_size
        self.loss2_sum += loss2.item() * b_x_size

        pre_lab1 = torch.argmax(output1, dim=1)
        # pre_lab2 = torch.argmax(sof(output2), dim=1)
        self.corrects1 += torch.sum(pre_lab1 == b_y1.data)
        # self.corrects2 += torch.sum(pre_lab2 == b_y2.data)
        self.data_num += b_x_size

        self.y1_true = torch.cat((self.y1_true, b_y1), dim=0)
        self.y1_pred = torch.cat((self.y1_pred, pre_lab1), dim=0)
        # self.y2_true = torch.cat((self.y2_true, b_y2), dim=0)
        # self.y2_pred = torch.cat((self.y2_pred, pre_lab2), dim=0)

        #self.y1_score = torch.cat((self.y1_score, sof(output1)), dim=0)
        #self.y2_score = torch.cat((self.y2_score, sof(output2)), dim=0)

    def var_update_all(self, variants,flag):
        if flag == 'train':

            self.train_loss_all.append(variants.loss)
            self.train_loss1_all.append(variants.loss1)
            self.train_loss2_all.append(variants.loss2)
            self.train_acc1_all.append(variants.acc1)
            self.train_acc2_all.append(variants.acc2)

            self.train_f1_c_all.append(variants.f1_c)
            self.train_f1_c_micro_all.append(variants.f1_micro_c)
            self.train_f1_c_macro_all.append(variants.f1_macro_c)
            self.train_f1_lvo_micro_all.append(variants.f1_micro_lvo)
            self.train_f1_lvo_macro_all.append(variants.f1_macro_lvo)

            self.train_precision_c_macro_all.append(variants.precision_macro_c)
            self.train_precision_c_micro_all.append(variants.precision_micro_c)

            self.train_recall_c_macro_all.append(variants.recall_macro_c)
            self.train_recall_c_micro_all.append(variants.recall_micro_c)

            self.train_specificity_c_all.append(variants.specificity_c)

        elif flag == 'test':

            self.test_loss_all.append(variants.loss)
            self.test_acc1_all.append(variants.acc1)
            self.test_acc2_all.append(variants.acc2)

            if np.max(self.test_acc1_all) > self.best_test_acc1 :
                self.best_test_acc1 = np.max(self.test_acc1_all)
                self.best_test_y1_true = variants.y1_true
                self.best_test_y1_pred = variants.y1_pred

            # self.best_test_acc2 = np.max(self.test_acc2_all)

            self.test_f1_c_all.append(variants.f1_c)
            self.test_f1_c_micro_all.append(variants.f1_micro_c)
            self.test_f1_c_macro_all.append(variants.f1_macro_c)
            self.test_f1_lvo_micro_all.append(variants.f1_micro_lvo)
            self.test_f1_lvo_macro_all.append(variants.f1_macro_lvo)

            self.test_precision_c_macro_all.append(variants.precision_macro_c)
            self.test_precision_c_micro_all.append(variants.precision_micro_c)

            self.test_recall_c_macro_all.append(variants.recall_macro_c)
            self.test_recall_c_micro_all.append(variants.recall_micro_c)

            self.test_specificity_c_all.append(variants.specificity_c)

        # self.y1true = torch.cat((self.y1true, test_variants.y1_true), dim=0)
        # self.y2true = torch.cat((self.y1true, test_variants.y2_true), dim=0)
        # self.y1pred = torch.cat((self.y1true, test_variants.y1_pred), dim=0)
        # self.y2pred = torch.cat((self.y1true, test_variants.y2_pred), dim=0)
        # self.y1score = torch.cat((self.y1score, sof(test_variants.y1_pred)), dim=0)
        # self.y2score = torch.cat((self.y2score, sof(test_variants.y2_pred)), dim=0)
    def var_calculate(self):
        """
        calculate acc and confusion
        """
        self.acc1 = self.corrects1.double().item() / self.data_num
        # self.acc2 = self.corrects2.double().item()/ self.data_num
        self.loss = self.loss_sum / self.data_num
        self.loss1 = self.loss1_sum / self.data_num
        self.loss2 = self.loss2_sum / self.data_num

        # self.f1_c = f1s(self.y1_true,self.y1_pred)
        self.f1_micro_c = f1s(self.y1_true, self.y1_pred, average='micro')
        self.f1_macro_c = f1s(self.y1_true, self.y1_pred, average='macro')
        # self.f1_micro_lvo = f1s(self.y2_true, self.y2_pred, average='micro')
        # self.f1_macro_lvo = f1s(self.y2_true, self.y2_pred, average='macro')

        self.precision_macro_c = precision_score(self.y1_true,self.y1_pred,average='macro')
        self.precision_micro_c = precision_score(self.y1_true, self.y1_pred, average='micro')

        self.recall_macro_c = recall_score(self.y1_true,self.y1_pred,average='macro')
        self.recall_micro_c = recall_score(self.y1_true, self.y1_pred, average='micro')
        #
        # self.specificity_c = specificity(self.y1_pred,self.y1_true)

