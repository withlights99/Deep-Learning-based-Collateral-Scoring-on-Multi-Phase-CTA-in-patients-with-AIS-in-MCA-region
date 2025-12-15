import numpy as np
import pandas as pd
import os
import shutil
from rigister_func import *
import SimpleITK as sitk   #导入SimpleElastix自定义的SimpleITK而不是标准库SimpleITK
import nibabel as nib

def rigister_exe1(fix_path,mov_path,save_path,population):
    parameterMapVector = sitk.VectorOfParameterMap()
    transmap = sitk.GetDefaultParameterMap("translation")

    rigidmap = sitk.GetDefaultParameterMap('rigid')
    # rigidmap['Transform'] = ['SimilarityTransform']
    affmap = sitk.GetDefaultParameterMap("affine")
    #bmap = sitk.GetDefaultParameterMap("bspline")

    transmap['ResampleInterpolator'] = ['FinalLinearInterpolator']
    transmap['Metric'] = ['AdvancedNormalizedCorrelation']

    rigidmap['ResampleInterpolator'] = ['FinalLinearInterpolator']
    rigidmap['Metric'] = ['AdvancedNormalizedCorrelation']

    rigidmap['MaximumNumberOfIterations'] = ['1024']

    affmap['ResampleInterpolator'] = ['FinalLinearInterpolator']
    affmap['Metric'] = ['AdvancedNormalizedCorrelation']
    affmap['MaximumNumberOfIterations'] = ['512']

    # bmap['ResampleInterpolator'] = ['FinalLinearInterpolator']
    # bmap['Metric'] = ['AdvancedNormalizedCorrelation']
    # bmap['MaximumNumberOfIterations'] = ['1024']
    parameterMapVector.append(transmap)
    parameterMapVector.append(rigidmap)
    parameterMapVector.append(affmap)
    parameterMapVector.append(rigidmap)
    parameterMapVector.append(affmap)
    # parameterMapVector.append(bmap)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage(fix_path))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(mov_path))
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.LogToConsoleOff()

    elastixImageFilter.Execute()

    img = sitk.Cast(elastixImageFilter.GetResultImage(),sitk.sitkInt16)
    sitk.WriteImage(img,os.path.join(save_path,'mCTA1.nii.gz'))

    
    
    
    transmap = elastixImageFilter.GetTransformParameterMap()
    transmap[0]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
    transmap[1]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
    transmap[2]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
    transmap[3]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
    transmap[4]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
    # transmap[5]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']

   
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transmap)
    
    for filename in population:
        if os.path.basename(filename) == 'NCCT_brain_interpolated.nii.gz':
            continue

        transformixImageFilter.SetMovingImage(sitk.ReadImage(filename))
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.Execute()
        img = sitk.Cast(transformixImageFilter.GetResultImage(),sitk.sitkInt16)
        sitk.WriteImage(img, os.path.join(save_path,os.path.basename(filename)))





def rigister_main(data_path,save_path):



    mov_path = os.path.join(data_path, 'mCTA1.nii.gz')
    fix_path = 'brain436_trilinear_processed.nii.gz'


    population = [os.path.join(data_path,'mCTA2.nii.gz'),
    os.path.join(data_path,'mCTA3.nii.gz')]

    rigister_exe1(fix_path,mov_path,save_path,population)
    print('rigistration done!')

    return

        
