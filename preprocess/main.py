import matlab.engine
import mask_use
import rigister_mcta
import clip
import intensity_scale
import mip_mcta
import colorviz

if __name__ == '__main__':
    # # ********** skull stripping based on MATLAB *********
    # # folder contains 3 phases CTA
    data_path = '/mnt/nvme1/data3/trash/PRoveIT-01-002'
    #
    # # matlab.engine        matlab R2023a and python3.8
    eng = matlab.engine.start_matlab()
    #
    # # save path of skullstripped data
    save_path = eng.ssmain_simple(data_path)
    #
    #
    # #*************** rigid rigistration *************
    #
    rigister_mcta.rigister_main(data_path,data_path)
    #
    #
    # #************* mask process ******************
    #
    # # first
    mask_use.mask_use(data_path,data_path,'mask436_processed_eroded_filled.nii.gz')
    rigister_mcta.rigister_main(data_path, data_path)
    mask_use.mask_use(data_path, data_path, 'mask436_processed_eroded_filled.nii.gz')
    #
    # MCA

    mask_use.mask_use(data_path,data_path,'MCA_mask_trilinear_eroded_filled.nii.gz')




    # ***************** intensity process and MIP ****************

    # intensity
    data_path = '/mnt/nvme1/data3/trash/PRoveIT-01-002'
    clip.clipd(data_path,data_path,450.0)
    # intensity_scale.intensity_scale(data_path,data_path,450.0)



    # MIP
    mip_mcta.mip_main(data_path,data_path)


    #***** colorviz *****

    colorviz.colorviz(data_path,data_path)
