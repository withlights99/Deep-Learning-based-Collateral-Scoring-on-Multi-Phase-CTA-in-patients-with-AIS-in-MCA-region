import os

import numpy as np
import nibabel as nib


def multilayer_mip(image, layer_num=32, axis=2):
    split_layers = np.array_split(image, layer_num, axis=axis)
    mip_list = [
        np.max(layer, axis=axis)
        for layer in split_layers
    ]
    mip_image = np.stack(mip_list, axis=axis)
    return mip_image


def overlap_slice_mip(img,thickness,intervals):
    i = 0
    split_layers = []
    while True :
        split_layers.append(img[:,:,i:i+thickness])
        i = i + intervals
        if i+intervals < np.shape(img) [2]:
            continue
        else:
            split_layers.append(img[:,:,i:])
            break
    axis = 2
    mip_list = [
        np.max(layer, axis=axis)
        for layer in split_layers
    ]
    mip_image = np.stack(mip_list, axis=axis)
    return mip_image

def mip_main(data_path,save_path):
    thickness = 32
    intervals = 8
    population = ['mCTA1.nii.gz', 'mCTA2.nii.gz', 'mCTA3.nii.gz']

    for basename in population:

        ni_image_path = os.path.join(data_path,basename)
        save_path = os.path.join(data_path,basename)

        image = nib.load(ni_image_path)
        ni_image = image.get_fdata()
        affine = image.affine

        ni_image = ni_image[:,:,160:320]
        # thresh = np.percentile(ni_image, 99.5)
        #thresh,ni_image = cv2.threshold(ni_image,thresh,300,cv2.THRESH_TOZERO_INV)
        #ni_image = np.clip(ni_image, 0, thresh)
        ni_image = np.max(ni_image, axis=2)

        # ni_image = overlap_slice_mip(ni_image,80,20)


        ni_image = nib.Nifti1Image(ni_image,affine)
        nib.save(ni_image,save_path)

