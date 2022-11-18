
import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import cv2

np.set_printoptions(threshold=np.inf)

relative_path_enable = 1

## Download Data
## wget https://github.com/tianrking/tianrking.github.io/releases/download/nii_data/data_source.zip
## unzip data_source.zip -d ../nii_data/

if not relative_path_enable:
    filepath = r'D:\邓SJ\chengang\seg_thor\data\data_source\Patient_01'  # 读取本代码同个文件夹下所有的nii格式的文件
    filenames = os.listdir(filepath)
    imgfile = r'D:\邓SJ\chengang\seg_thor\data\data_source\Patient_01'
else:
    filepath = "../nii_data/"
    absolute_path = "/home/w0x7ce/Desktop/AQ/nii_data"



class PG:
    
    def __init__(self) -> None:
        self.Patient = ""
        self.GT = ""
        self.Name = ""
        
    def Print(self) -> None:
        print(self.Patient)
        print(self.GT)
        
for kk in range(10,40):
    
    filenames = filepath + "Patient_%s"%"02"

    _PG = PG()
    _name = ""
    for index,name in  enumerate(os.listdir(filenames)):
        
        if name.startswith('Patient') and name.endswith('nii'):
                
            _PG.Patient = name
            _PG.Name = _PG.Patient.split(".")[0]
            print(_PG.Name)
            print(_PG.Patient)
            img = nib.load(absolute_path + "/" + _PG.Name + "/" + _PG.Patient)  # 读取nii
            data = img.get_fdata()
            print(data.shape,type(data))
            
            if not os.path.exists("DATA_IGNORE_FILE/%s"%_PG.Name):
                os.mkdir("DATA_IGNORE_FILE/%s"%_PG.Name)
                os.mkdir("DATA_IGNORE_FILE/%s/%s"%(_PG.Name,"Patient"))
                os.mkdir("DATA_IGNORE_FILE/%s/%s"%(_PG.Name,"GT"))

            for i in range(0,data.shape[2]):
                slice_2 = data[:, :, i]
                cv2.imwrite('DATA_IGNORE_FILE/%s/Patient/%s_mask.png'%(_PG.Name,i),slice_2)

        # if name.startswith('GT') and name.endswith('nii'):
            
        #     _PG.Name = "Patient_%s" % kk
        #     _PG.GT = name
            
        #     img = nib.load(absolute_path + "/" + _PG.Name + "/" + _PG.GT)
        #     data = img.get_fdata()
        #     print(data.shape,type(data))
            
        #     if not os.path.exists("DATA_IGNORE_FILE/%s"%_PG.Name):
        #         os.mkdir("DATA_IGNORE_FILE/%s"%_PG.Name)
        #         os.mkdir("DATA_IGNORE_FILE/%s/%s"%(_PG.Name,"Patient"))
        #         os.mkdir("DATA_IGNORE_FILE/%s/%s"%(_PG.Name,"GT"))

        #     for i in range(0,data.shape[2]):
        #         slice_2 = data[:, :, i]
        #         cv2.imwrite('DATA_IGNORE_FILE/%s/GT/%s_mask.png'%(_PG.Name,i),slice_2)

        print(str(index) + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    
    exit
        