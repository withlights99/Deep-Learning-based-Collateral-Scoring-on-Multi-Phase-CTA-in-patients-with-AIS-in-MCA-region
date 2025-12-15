import os
import numpy as np
import nibabel as nib


def intensity_scale(data_path,save_path,threshold):

    population = ['mCTA1.nii.gz', 'mCTA2.nii.gz', 'mCTA3.nii.gz']


    for basename in population:

        ni_image_path = os.path.join(data_path,basename)


        image = nib.load(ni_image_path)
        ni_image = image.get_fdata()
        affine = image.affine


        ni_image = (ni_image)/threshold
        ni_image = nib.Nifti1Image(ni_image,affine)
        nib.save(ni_image,os.path.join(save_path,basename))
        # print(ori_dir+'Process done!')
