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
from monai.transforms import Compose,LoadImaged,EnsureChannelFirstd,ToTensord,Resized,ScaleIntensityRanged,ConcatItemsd,RandFlipd,ScaleIntensityd,SqueezeDimd
import random
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score,roc_auc_score,cohen_kappa_score,brier_score_loss
from sklearn.metrics import f1_score as f1s
from sklearn.calibration import calibration_curve
import codecs


from model import FC2SENet

from train_func import *
import sys 
from roc_auc import ROC_paint,ROC_2
from scipy import stats
import initial

import vision_func
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold,StratifiedKFold
import pingouin as pg
from mass import calculate_kappa as kappa

#sys.path.insert(0, '../build/SimpleITK-build/Wrapping/Python/SimpleITK.py')
#import SimpleITK as sitk

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

train_recording = {}
test_recording = {}
def train_model(model,traindataloader,testdataloader,lr,criterion1,optimizer,epochs=50):
    best_model_wts = copy.deepcopy(model.state_dict())
    sof = nn.Softmax(dim=1)
    since = time.time()
    #writer = SummaryWriter(log_dir='/home/liuhao/train/code2/train_main/outcome_vision')

    variants = initial.variants_init('global')


    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        train_variants = initial.variants_init('train')



        for step,batch_data in enumerate(traindataloader):
            b_x,b_y1,name= batch_data["inputs"].to(device),batch_data["c_label"].to(device),batch_data["name"]
            model.train()
            optimizer.zero_grad()

            output1 = model(b_x)
            output2 = output1
            # print(sof(output1))

            x = sof(output1)
            x = torch.argmax(x, dim=1)

            for i in range(len(x)):
                if name[i] not in train_recording:
                    train_recording[name[i]] = 0
                if x[i] == b_y1[i]:
                    train_recording[name[i]] += 1

            loss1 = criterion1(output1, b_y1.long())
            # loss2 = criterion1(output2, b_y2.long())
            loss2 = loss1
            loss = loss1

            train_variants.var_updata(loss.to('cpu'),output1.to('cpu'),output2.to('cpu'),b_x.size(0),b_y1.to('cpu'),b_y2.to('cpu'),loss1.to('cpu'),loss2.to('cpu'),name)

            loss.backward()
            optimizer.step()

        train_variants.var_calculate()
        # writer.add_scalars('loss/train', {'loss1':train_variants.loss1,'loss2':train_variants.loss2,'loss': train_variants.loss}, epoch)
        # writer.add_scalars('acc/train', {'acc1': train_variants.acc1, 'acc2': train_variants.acc2},epoch)
        print("train_loss:{}".format(train_variants.loss))

        # if np.array(train_variants.loss) < np.array(variants.least_train_loss): #************************************
        #     print(1)
        #     variants.least_train_loss = train_variants.loss
        #     best_model_wts = copy.deepcopy(model.state_dict())
        variants.var_update_all(train_variants,'train')

        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(
            time_use // 60, time_use % 60
        ))
        lr.step()
    # *************** test **************************
        # model.load_state_dict(best_model_wts)
        test_variants = initial.variants_init('test')
        with torch.no_grad(): #test
            for batch_data in testdataloader:
                b_x, b_y1, b_y2,test_name = batch_data["inputs"].to(device), batch_data["c_label"].to(device),batch_data["name"]
                model.eval()
                output1 = model(b_x)

                # print(sof(output1))

                x = sof(output1)
                x = torch.argmax(x, dim=1)

                for i in range(len(x)):
                    if test_name[i] not in test_recording:
                        test_recording[test_name[i]] = 0
                    if x[i] == b_y1[i]:
                        test_recording[test_name[i]] += 1




                output2 =output1
                loss1 = criterion1(output1, b_y1.long())
                loss2 = loss1
                loss = loss1


                test_variants.var_updata(loss.to('cpu'), output1.to('cpu'), output2.to('cpu'), b_x.size(0), b_y1.to('cpu'), b_y2.to('cpu'),loss1.to('cpu'),loss2.to('cpu'),test_name)
            test_variants.var_calculate()

        mod = variants.var_update_all(test_variants,'test')
        if mod == 1:
            best_model_wts = copy.deepcopy(model.state_dict())

    # writer.add_scalars('loss/test', {'loss1': test_variants.loss1, 'loss2': test_variants.loss2, 'loss': test_variants.loss},epoch)
    # writer.add_scalars('acc/test', {'acc1': test_variants.acc1, 'acc2': test_variants.acc2}, epoch)
    # vision_func.out_print(epoch,variants)
        print('在测试集上的acc1为{}'.format(test_variants.acc1))
        # print('在测试集上的acc2为{}'.format(test_variants.acc2))



    train_process = pd.DataFrame(
        data={
            "epoch":range(epochs),
            "train_loss_all":variants.train_loss_all,
            "train_loss1_all": variants.train_loss1_all,
            "train_loss2_all": variants.train_loss2_all,
            "train_f1_c_micro_all":variants.train_f1_c_micro_all,
            "train_f1_lvo_micro_all": variants.train_f1_lvo_micro_all,
            "train_f1_c_macro_all": variants.train_f1_c_macro_all,
            "train_f1_lvo_macro_all": variants.train_f1_lvo_macro_all,
            "train_acc1_all":variants.train_acc1_all,
            "train_acc2_all":variants.train_acc2_all,

            # "test_loss_all":variants.test_loss_all,
            # "test_f1_c_micro_all":variants.test_f1_c_micro_all,
            # "test_acc1_all":variants.test_acc1_all,
            # "test_f1_lvo_micro_all": variants.test_f1_lvo_micro_all,
            # "test_acc2_all":variants.test_acc2_all,

        }
    )
    model.load_state_dict(best_model_wts)

    # writer.close()
    return  model,train_process,variants,test_variants



# ******transforms*******
spa_size = 128
train_transform = Compose(
    [
        LoadImaged(keys=['image1','image2','image3']),
        # ScaleIntensityRanged(keys=['image1','image2','image3'], a_min=0, a_max=350, b_min=0.0, b_max=1.0, clip=True, dtype=np.float32),
        #ScaleIntensityd(keys='image',minv=0.0, maxv=1.0),

        EnsureChannelFirstd(keys=['image1','image2','image3']),
        RandFlipd(keys=['image1','image2','image3'],spatial_axis=1 ,prob=0.4),
        # SqueezeDimd(keys=['image1','image2','image3'],dim=2),
        #Resized(keys=['image'], spatial_size=(spa_size, spa_size, spa_size), size_mode='all'),

        # RandRotated(keys=['image1','image2','image3'], range_x=0.1,prob=0.5),
        #RandAffined(keys=['image', 'lab1', 'lab2'],prob=0.5,translate_range=5),
        ConcatItemsd(keys=['image1','image2','image3'], name='inputs'),
        ToTensord(keys=['image1','image2','image3','inputs'])
    ]
)
transform = Compose(
    [
        LoadImaged(keys=['image1','image2','image3']),
        # ScaleIntensityRanged(keys=['image1', 'image2', 'image3'], a_min=0, a_max=350, b_min=0.0, b_max=1.0, clip=True,dtype=np.float32),
        #ScaleIntensityRanged(keys=['image'], a_min=-20, a_max=350, b_min=0.0, b_max=1.0, clip=True, dtype=np.float32),
        #SqueezeDimd(keys='image',dim=2),
        #ScaleIntensityRanged(keys=['image'], a_min=-20, a_max=400, b_min=0.0, b_max=1.0, clip=True, dtype=np.float32),

        EnsureChannelFirstd(keys=['image1','image2','image3']),
        # SqueezeDimd(keys=['image1', 'image2', 'image3'],dim=2),

        # Resized(keys=['image'], spatial_size=(spa_size, spa_size, spa_size), size_mode='all'),
        ConcatItemsd(keys=['image1','image2','image3'], name='inputs'),
        ToTensord(keys=['image1','image2','image3','inputs'])
    ]
)



if __name__ == '__main__':

    # *********** excel reading，creating scores list **************
    excel_path = ''
    name_scores = initial.cre_names_scores(excel_path)

    # dict loading
    data_path = '/mnt/nvme1/data3/ss_rpm436_smip_mcta'
    data_dict = initial.crea_dict(data_path, name_scores)



    # *****dict classification****
    data_d0 = []
    data_d1 = []
    data_d2 = []
    for dict in data_dict:
        if dict['c_label'] == 0:
            data_d0.append(dict)
        elif dict['c_label'] == 1:
            data_d1.append(dict)
        else:
            data_d2.append(dict)
    print(len(data_d0))
    print(len(data_d1))
    print(len(data_d2))


    data_dict = data_d0 + data_d1 +data_d2
    random.shuffle(data_dict)

    scores = []
    for data in data_dict:
        scores.append(data['c_label'])

    print([len(data_dict),len(scores)])

    KF = StratifiedKFold(n_splits=5,shuffle=True)
    metrics = {
        "acc":[],
        "f1_macro":[],
        "f1_micro": [],
        "recall_macro":[],
        "recall_micro":[],
        "precision_macro":[],
        "precision_micro":[],
        "specificity":[],
        "ROC":[],
        "kappa": [],
        "icc": [],
    }
    # acc1_sum = 0.0
    # acc2_sum = 0.0
    # f1score_macro_sum = 0.0
    # f1score_micro_sum = 0.0
    # recall_macro_sum = 0.0
    # recall_micro_sum = 0.0
    # precision_macro_sum = 0.0
    # precision_micro_sum = 0.0
    # specificity_macro_sum = 0.0
    # specificity_micro_sum = 0.0
    num = 1
    cross_y_true = torch.tensor([],dtype=torch.int)
    cross_y_pred = torch.tensor([],dtype=torch.int)
    cross_y_score = torch.tensor([], dtype=torch.int)
    cross_name = []

    for train_index, test_index in KF.split(data_dict,scores):
        train_dict, test_dict = [data_dict[x] for x in train_index], [data_dict[x] for x in test_index]


        print('***************fold {}***************'.format(num))
        # ******dataloader*******
        with open('/home/liuhao/train/code2/train_main/train_dict_{}.txt'.format(num), 'w', encoding='utf-8') as f:
            for data in train_dict:

                f.write(data['name']+'\n')
        with open('/home/liuhao/train/code2/train_main/test_dict_{}.txt'.format(num), 'w', encoding='utf-8') as f:
            for data in test_dict:

                f.write(data['name']+'\n')

        random.shuffle(train_dict)
        random.shuffle(test_dict)

        train_ds = Dataset(data=train_dict, transform=train_transform)
        test_ds = Dataset(data=test_dict, transform=transform)
        train_loader = monai.data.DataLoader(
            train_ds, batch_size=8, shuffle=True, num_workers=2
        )

        test_loader = monai.data.DataLoader(
            test_ds, batch_size=4, num_workers=2
        )

        myconvnet = FC2SENet(spatial_dims=2,in_channels=3,out_channels=3,dropout_prob=0.5).to(device)


        optimizer = Adam(myconvnet.parameters(),lr=5e-5)#,weight_decay=0.001

        exp_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,50,70],gamma=0.32,last_epoch=-1)
        criterion1 = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.0,16.0  ]).to(device))#weight=torch.tensor([1.0,2.4,9]).to(device)
        myconvnet,train_process,variants,test_variants = train_model(
            myconvnet,train_loader,test_loader,exp_lr,criterion1,optimizer,epochs=75
        )
        # torch.save(myconvnet,'./model/{}.pth'.format(num))
        paint_path ='/home/liuhao/train/code2/train_main/outcome_{}'.format(num)

        vision_func.all_paint_kfold(variants,paint_path)
        cross_y_true = torch.cat((cross_y_true,variants.best_test_y1_true),dim=0)
        cross_y_pred = torch.cat((cross_y_pred,variants.best_test_y1_pred),dim=0)
        cross_y_score = torch.cat((cross_y_score, variants.best_test_y1_score), dim=0)
        cross_name = cross_name + variants.best_test_name

        metrics["acc"].append(variants.best_test_acc1)
        metrics["f1_macro"].append(f1s(variants.best_test_y1_true,variants.best_test_y1_pred,average='macro'))
        metrics["f1_micro"].append(f1s(variants.best_test_y1_true, variants.best_test_y1_pred, average='micro'))
        metrics["recall_macro"].append(recall_score(variants.best_test_y1_true, variants.best_test_y1_pred, average='macro'))
        metrics["recall_micro"].append(recall_score(variants.best_test_y1_true, variants.best_test_y1_pred, average='micro'))
        metrics["precision_macro"].append(precision_score(variants.best_test_y1_true, variants.best_test_y1_pred, average='macro'))
        metrics["precision_micro"].append(precision_score(variants.best_test_y1_true, variants.best_test_y1_pred, average='micro'))
        metrics["specificity"].append(
            initial.cal_metrics(confusion_matrix(variants.best_test_y1_true, variants.best_test_y1_pred)))
        metrics["ROC"].append(
            roc_auc_score(variants.best_test_y1_true,variants.best_test_y1_score,average='macro',multi_class='ovr'))
        print(type(variants.best_test_y1_true))
        print(type(variants.best_test_y1_score))
        metrics["kappa"].append(
            cohen_kappa_score(variants.best_test_y1_true, variants.best_test_y1_pred))
        metrics["icc"].append(
            cal_icc(variants.best_test_y1_true, variants.best_test_y1_pred)['ICC'][0])
        icc1 =cal_icc(variants.best_test_y1_true, variants.best_test_y1_pred)['ICC'][0]
        # print(icc1)
        print(metrics)

        num += 1

    confu_path = '/home/liuhao/train/code2/train_main/test_confusion_c'






    train_confusion(cross_y_true, cross_y_pred, confu_path)
    ROC_paint(cross_y_true, cross_y_score, '3class_roc')
    print('acc:{}'.format(calculate_confidence_interval(metrics["acc"])))
    print('f1:{}'.format([calculate_confidence_interval(metrics["f1_macro"]),calculate_confidence_interval(metrics["f1_micro"])]))
    print('recall:{}'.format([calculate_confidence_interval(metrics["recall_macro"]),calculate_confidence_interval(metrics["recall_micro"])]))
    print('precision:{}'.format([calculate_confidence_interval(metrics["precision_macro"]),calculate_confidence_interval(metrics["precision_micro"])]))
    print('ROC:{}'.format([calculate_confidence_interval(metrics["ROC"])]))
    print('specificity:{}'.format(calculate_confidence_interval(metrics["specificity"])))
    print('kappa:{}'.format(calculate_confidence_interval(metrics["kappa"])))
    print('icc:{}'.format(calculate_confidence_interval(metrics["icc"])))

    roc_mean = np.mean(metrics["ROC"])
    print(metrics["ROC"])
    t,p = stats.ttest_1samp(np.array(metrics["ROC"]),roc_mean)
    print("AUC:t,p   {}".format([t,p]))


    vision_func.recording_paint(train_recording,'/home/liuhao/train/code2/train_main/train_recording')
    vision_func.recording_paint(test_recording,'/home/liuhao/train/code2/train_main/test_recording')


    classd = ['good','intermediate','poor']
    plt.figure()
    for i in range(cross_y_score.shape[1]):
        prob_true, prob_pred = calibration_curve((cross_y_pred == i).astype(int), cross_y_score[:, i], n_bins=10)

        plt.plot(prob_pred, prob_true, marker='o', label=f'Class {classd[i]}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration Curves per Class')
    plt.legend()
    plt.show()
    plt.savefig('Calibration')








