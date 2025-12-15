import os
import numpy as np
import nibabel as nib
import shutil

from PIL import Image


def colorviz(data_path,save_path):


    data1 = nib.load(data_path + '/mCTA1.nii.gz').get_fdata().squeeze()
    data2 = nib.load(data_path + '/mCTA2.nii.gz').get_fdata().squeeze()
    data3 = nib.load(data_path + '/mCTA3.nii.gz').get_fdata().squeeze()

    data1 = Image.fromarray((data1[60:376,60:376] * 255).astype(np.uint8)).transpose(Image.ROTATE_90)
    data2 = Image.fromarray((data2[60:376,60:376] * 255).astype(np.uint8)).transpose(Image.ROTATE_90)
    data3 = Image.fromarray((data3[60:376,60:376] * 255).astype(np.uint8)).transpose(Image.ROTATE_90)

    depth = 316
    data = Image.merge('RGB', (data1, data2, data3))
    data1.save(os.path.join(data_path,'mCTA1.png'))
    data2.save(os.path.join(data_path, 'mCTA2.png'))
    data3.save(os.path.join(data_path, 'mCTA3.png'))
    data.save(os.path.join(data_path, 'colorviz.png'))
