
function save_path=ssmain_simple(data_path)
import SkullStripSingleCT_MF.*
file = data_path
name1 = strcat(file,'/mCTA1.nii.gz')
SkullStripSingleCT_MF(name1);
name2 = strcat(file,'/mCTA2.nii.gz')
SkullStripSingleCT_MF(name2);
name3 = strcat(file,'/mCTA3.nii.gz')
SkullStripSingleCT_MF(name3);
disp([file '----skull tripping finished'])
%movefile(file,'/mnt/nvme1/data4/')
save_path = data_path

end
