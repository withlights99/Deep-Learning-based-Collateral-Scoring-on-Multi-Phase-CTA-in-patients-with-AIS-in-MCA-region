
function y = strip_skull(pname)
%img_path = strcat(pname,'/mCTA2.nii.gz');
img_path = pname
brain_path = img_path;
addpath('NIfTI_20140122');

pathNCCTImage = [img_path];
PathNCCT_Brain = [brain_path];

%disp(['Strip skull of patient ' pathNCCTImage]);

% load the subject image
ImgSubj_nii = load_untouch_nii(pathNCCTImage);
ImgSubj_hdr = ImgSubj_nii.hdr;
ImgSubj = ImgSubj_nii.img;
%ImgSubj = double(ImgSubj);

% skull stripping
NCCT_Thr = 100; % for NCCT images
CTA_Thr = 400; % for CTA images

[brain] = SkullStripping(double(ImgSubj),CTA_Thr);

% save image
Output_nii.hdr = ImgSubj_hdr;
Output_nii.img = int16(brain);
save_nii(Output_nii, PathNCCT_Brain);

%disp([pathNCCTImage '----skull tripping finished']);
clear;
clc;
close all;
end

