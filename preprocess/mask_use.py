import os
import numpy as np
import nibabel as nib

def mask_use(data_path,save_path,mask_path):

	population = ['mCTA1.nii.gz', 'mCTA2.nii.gz', 'mCTA3.nii.gz' ]
	for basename in population:
		ni_image_path = os.path.join(data_path, basename)

		image = nib.load(ni_image_path)
		ni_image = image.get_fdata()
		affine = image.affine
		mask = nib.load(mask_path).get_fdata()
		# mask = np.where(mask==0,1,0)

		ni_image = ni_image * mask


		ni_image = nib.Nifti1Image(ni_image, affine)
		nib.save(ni_image, os.path.join(save_path,basename))
	print(data_path + 'Process done!')

