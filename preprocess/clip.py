import os
import numpy as np
import nibabel as nib


def clipd(data_path,save_path,threshold):

    population = ['mCTA1.nii.gz', 'mCTA2.nii.gz', 'mCTA3.nii.gz']

    for basename in population:

        ni_image_path = os.path.join(data_path,basename)


        image = nib.load(ni_image_path)
        ni_image = image.get_fdata()
        affine = image.affine

        ni_image = ni_image[:,:,:]
        #thresh = np.percentile(ni_image, 99.999)
        # thresh,ni_image = cv2.threshold(ni_image,500,500,cv2.THRESH_TOZERO_INV)
        # thresh, ni_image = cv2.threshold(ni_image, 500, 500, cv2.THRESH_TOZERO_INV)
        ni_image = np.clip(ni_image, 0, threshold)
        ni_image = (ni_image) / threshold
        #ni_image = np.max(ni_image, axis=2)



        ni_image = nib.Nifti1Image(ni_image,affine)
        nib.save(ni_image,os.path.join(save_path,basename))
        # print('clip Process done!')
