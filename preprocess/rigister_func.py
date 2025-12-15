import numpy as np
import pandas as pd
import os
import shutil
import math
import SimpleITK as sitk   #导入SimpleElastix自定义的SimpleITK而不是标准库SimpleITK

def rigister_exe(fix_path,mov_path,save_path,population):
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("translation"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    parameterMapVector[0]['ResampleInterpolator'] = ['FinalLinearInterpolator']
    parameterMapVector[1]['ResampleInterpolator'] = ['FinalLinearInterpolator']
    parameterMapVector[2]['MaximumNumberOfIterations'] = ['512']
    parameterMapVector[2]['ResampleInterpolator'] = ['FinalLinearInterpolator']
    parameterMapVector[2]['Metric'] = ['AdvancedNormalizedCorrelation']

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage(fix_path))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(mov_path))
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    img = sitk.Cast(elastixImageFilter.GetResultImage(),sitk.sitkInt8)
    sitk.WriteImage(img,save_path+"/rbrain.nii.gz")

    #for filename in population:
    #    transformixImageFilter.SetMovingImage(sitk.ReadImage(filename))
     #   transformixImageFilter.Execute()
      #  img = sitk.Cast(transformixImageFilter.GetResultImage(),sitk.sitkInt8)
       # sitk.WriteImage(img, save_path+'/'+os.path.basename(filename))


def rigister_mask(data_path,save_path,mask_path):
    itk_img = sitk.ReadImage(data_path)
    img_array = sitk.GetArrayFromImage(itk_img)
    origin =itk_img.GetOrigin()
    direction = itk_img.GetDirection()
    space = itk_img.GetSpacing()

    itk_mask = sitk.ReadImage(mask_path)
    itk_maska = sitk.GetArrayFromImage(itk_mask)
    img_array= img_array * itk_maska

    itk_img = sitk.GetImageFromArray(img_array)
    itk_img.SetOrigin(origin)
    itk_img.SetDirection(direction)
    itk_img.SetSpacing(space)
    sitk.WriteImage(itk_img, save_path)

def rigister_f2m(fix_path,mov_path,save_path,population):
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("translation"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector[0]['ResampleInterpolator'] = ['FinalLinearInterpolator']
    parameterMapVector[1]['ResampleInterpolator'] = ['FinalLinearInterpolator']
    parameterMapVector[0]['Metric'] = ['AdvancedNormalizedCorrelation']
    parameterMapVector[1]['Metric'] = ['AdvancedNormalizedCorrelation']

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage(fix_path))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(mov_path))
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    img = sitk.Cast(elastixImageFilter.GetResultImage(),sitk.sitkInt16)
    sitk.WriteImage(img,save_path+"/mCTA2.nii.gz")

    transmap = elastixImageFilter.GetTransformParameterMap()
    transmap[0]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
    transmap[1]['ResampleInterpolator'] = ['FinalNearestNeighborInterpolator']
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transmap)

    for filename in population:
        transformixImageFilter.SetMovingImage(sitk.ReadImage(filename))
        transformixImageFilter.Execute()
        img = sitk.Cast(transformixImageFilter.GetResultImage(),sitk.sitkInt8)
        sitk.WriteImage(img, save_path+'/'+os.path.basename(filename))

def Interpolation3D(src_img, dst_size):
    #srcImage原3d图像，dst_size插值后图像的大小
    srcZ, srcY, srcX = src_img.shape   #原图图像大小
    dst_img = np.zeros(shape = dst_size, dtype = np.int16)  #插值后的图像

    new_Z, new_Y, new_X = dst_img.shape  #
    print("插值后图像的大小", dst_img.shape)

    factor_z = srcZ / new_Z
    factor_y = srcY / new_Y
    factor_x = srcX / new_X

    for z in range(new_Z):
        for y in range(new_Y):
            for x in range(new_X):

                src_z = z * factor_z
                src_y = y * factor_y
                src_x = x * factor_x

                src_z_int = math.floor(z * factor_z)
                src_y_int = math.floor(y * factor_y)
                src_x_int = math.floor(x * factor_x)

                w = src_z - src_z_int
                u = src_y - src_y_int
                v = src_x - src_x_int

                # 判断是否查出边界
                if src_x_int + 1 == srcX or src_y_int + 1 == srcY or src_z_int + 1 == srcZ:
                    dst_img[z, y , x] = src_img[src_z_int, src_y_int, src_x_int]

                else:

                    C000 = src_img[src_z_int, src_y_int, src_x_int]
                    C001 = src_img[src_z_int, src_y_int, src_x_int + 1]
                    C011 = src_img[src_z_int, src_y_int + 1, src_x_int + 1]
                    C010 = src_img[src_z_int, src_y_int + 1, src_x_int]
                    C100 = src_img[src_z_int + 1, src_y_int, src_x_int]
                    C101 = src_img[src_z_int + 1, src_y_int, src_x_int + 1]
                    C111 = src_img[src_z_int + 1, src_y_int + 1, src_x_int + 1]
                    C110 = src_img[src_z_int + 1, src_y_int + 1, src_x_int]
                    dst_img[z ,y ,x] = C000 * (1 - v)* (1 - u) * (1 - w) + \
                                       C100 * v * (1 - u) * (1 - w) + \
                                       C010 * (1- v) * u * (1 - w) + \
                                       C001 * (1 - v) * (1 - u) * w + \
                                       C101 * v * (1 - u) * w + \
                                       C011 * (1 - v) * u * w + \
                                       C110 * v * u * (1 - w) + \
                                       C111 * v * u * w
    print('interpolation is done!')
    return dst_img