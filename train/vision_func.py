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



def train_paint_kfold(train_process, savefig_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 2, 1)
    plt.plot(
        train_process.epoch, train_process.train_loss_all, "ro-", label="Train loss"
    )
    plt.plot(
        train_process.epoch, train_process.train_loss1_all, "bs-", label="train loss1"
    )
    plt.plot(
        train_process.epoch, train_process.train_loss2_all, "gs-", label="train loss2"
    )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(2, 2, 2)
    plt.plot(
        train_process.epoch, train_process.train_acc1_all, "ro-", label="Train acc1"
    )
    plt.plot(
        train_process.epoch, train_process.train_acc2_all, "bo-", label="Train acc2"
    )
    # plt.plot(
    #     train_process.epoch, train_process.test_acc1_all, "rs-", label="test acc1"
    # )
    # plt.plot(
    #     train_process.epoch, train_process.test_acc2_all, "bs-", label="test acc2"
    # )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.subplot(2, 2, 3)
    plt.plot(
        train_process.epoch, train_process.train_f1_c_micro_all, "ro-", label="Train f1_c_micro"
    )
    plt.plot(
        train_process.epoch, train_process.train_f1_c_macro_all, "bo-", label="Train f1_c_macro"
    )
    plt.plot(
        train_process.epoch, train_process.train_f1_lvo_micro_all, "rs-", label="Train f1_lvo_micro"
    )
    plt.plot(
        train_process.epoch, train_process.train_f1_lvo_macro_all, "bs-", label="Train f1_lvo_macro"
    )

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("f1_score")

    plt.savefig(savefig_path)
    # plt.show()

def test_paint_kfold(variants, savefig_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 2, 1)
    epoch = range(len(variants.test_loss_all))
    plt.plot(
        epoch, variants.test_loss_all, "ro-", label="test loss"
    )
    # plt.plot(
    #     epoch, train_process.train_loss1_all, "bs-", label="train loss1"
    # )
    # plt.plot(
    #     epoch, train_process.train_loss2_all, "gs-", label="train loss2"
    # )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(2, 2, 2)
    plt.plot(
        epoch, variants.test_acc1_all, "ro-", label="test acc1"
    )
    plt.plot(
        epoch, variants.test_acc2_all, "bo-", label="test acc2"
    )
    # plt.plot(
    #     train_process.epoch, train_process.test_acc1_all, "rs-", label="test acc1"
    # )
    # plt.plot(
    #     train_process.epoch, train_process.test_acc2_all, "bs-", label="test acc2"
    # )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    # plt.subplot(2, 2, 3)
    # plt.plot(
    #     train_process.epoch, train_process.train_f1_c_micro_all, "ro-", label="Train f1_c_micro"
    # )
    # plt.plot(
    #     train_process.epoch, train_process.train_f1_c_macro_all, "bo-", label="Train f1_c_macro"
    # )
    # plt.plot(
    #     train_process.epoch, train_process.train_f1_lvo_micro_all, "rs-", label="Train f1_lvo_micro"
    # )
    # plt.plot(
    #     train_process.epoch, train_process.train_f1_lvo_macro_all, "bs-", label="Train f1_lvo_macro"
    # )

    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("f1_score")

    plt.savefig(savefig_path)
    # plt.show()

def all_paint_kfold(variants, savefig_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 2, 1)
    epoch = range(len(variants.test_loss_all))
    plt.plot(
        epoch, variants.test_loss_all, "bo-", label="test loss"
    )
    plt.plot(
        epoch, variants.train_loss_all, "ro-", label="train loss"
    )
    # plt.plot(
    #     epoch, train_process.train_loss2_all, "gs-", label="train loss2"
    # )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(2, 2, 2)
    plt.plot(
        epoch, variants.test_acc1_all, "bo-", label="test acc"
    )
    plt.plot(
        epoch, variants.train_acc1_all, "ro-", label="train acc"
    )
    # plt.plot(
    #     train_process.epoch, train_process.test_acc1_all, "rs-", label="test acc1"
    # )
    # plt.plot(
    #     train_process.epoch, train_process.test_acc2_all, "bs-", label="test acc2"
    # )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")

    plt.subplot(2, 2, 3)
    plt.plot(
        epoch, variants.train_f1_c_macro_all, "ro-", label="Train f1_macro_c"
    )
    plt.plot(
        epoch, variants.train_f1_c_micro_all, "yo-", label="Train f1_micro_c"
    )
    plt.plot(
        epoch, variants.test_f1_c_macro_all, "bo-", label="Test f1_macro_c"
    )

    plt.plot(
        epoch, variants.test_f1_c_micro_all, "go-", label="Test f1_micro_c"
    )

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("f1_score")

    plt.subplot(2, 2, 4)
    plt.plot(
        epoch, variants.train_precision_c_macro_all, "ro-", label="Train precision_macro_c"
    )
    # plt.plot(
    #     epoch, variants.train_precision_c_micro_all, "rs-", label="Train precision_micro_c"
    # )
    plt.plot(
        epoch, variants.test_precision_c_macro_all, "bo-", label="Test precision_macro_c"
    )
    # plt.plot(
    #     epoch, variants.test_precision_c_micro_all, "bs-", label="Test precision_micro_c"
    # )
    plt.plot(
        epoch, variants.train_recall_c_macro_all, "go-", label="Train recall_macro_c"
    )
    # plt.plot(
    #     epoch, variants.train_recall_c_micro_all, "gs-", label="Train recall_micro_c"
    # )
    plt.plot(
        epoch, variants.test_recall_c_macro_all, "yo-", label="Test recall_macro_c"
    )
    # plt.plot(
    #     epoch, variants.test_recall_c_micro_all, "ys-", label="Test recall_micro_c"
    # )


    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("TFPN")


    plt.savefig(savefig_path)
    # plt.show()


def train_confusion(y_true, y_pred,confu_path):
    c = confusion_matrix(y_true, y_pred)
    plt.matshow(c, cmap=plt.cm.Reds)
    for i in range(len(c)):
        for j in range(len(c)):
            plt.annotate(c[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(confu_path)
    # plt.show()

def out_print(epoch, variants):

    print('{} Train Loss:{:.4f} Train Acc1: {:.4f} Train Acc2: {:.4f}'.format(
        epoch, variants.train_loss_all[-1], variants.train_acc1_all[-1], variants.train_acc2_all[-1]
    ))

    # print('{} Val Loss:{:.4f} Val Acc1: {:.4f} Val Acc2: {:.4f}'.format(
    #     epoch, variants.test_loss_all[-1], variants.test_acc1_all[-1],variants.test_acc2_all[-1]
    # ))
    print('{} test Loss:{:.4f} test Acc1: {:.4f} test Acc2: {:.4f}'.format(
        epoch, variants.test_loss_all[-1], variants.test_acc1_all[-1], variants.test_acc2_all[-1]
    ))
    print('{} train_f1_c_micro:{:.4f} test_f1_c_mirco: {:.4f}'.format(
        epoch, variants.train_f1_c_micro_all[-1], variants.test_f1_c_micro_all[-1]
    ))
    print('{} train_f1_2:{:.4f} Val_f1_2: {:.4f}'.format(
        epoch, variants.train_f1_lvo_micro_all[-1], variants.test_f1_lvo_micro_all[-1]
    ))


def recording_paint(recording,save_path):
    #******字典排序****
    recording = sorted(recording.items(),key = lambda x:x[1])
    recording = dict(recording[:30])
    plt.figure(figsize=(30, 30))
    plt.subplot(1, 1, 1)
    plt.plot(
        recording.keys(), recording.values(), "ro-", label="wrong number"
    )

    plt.legend()
    plt.xlabel("name")
    plt.ylabel("number")



    plt.savefig(save_path)
    # plt.show()













 





