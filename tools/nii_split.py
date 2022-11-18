
import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import cv2

np.set_printoptions(threshold=np.inf)

relative_path_enable = 1

## Download Data
## wget https://github.com/tianrking/tianrking.github.io/releases/download/nii_data/data_source.zip
## unzip data_source.zip -d 

if not relative_path_enable:
    filepath = r'D:\邓SJ\chengang\seg_thor\data\data_source\Patient_01'  # 读取本代码同个文件夹下所有的nii格式的文件
    filenames = os.listdir(filepath)
    imgfile = r'D:\邓SJ\chengang\seg_thor\data\data_source\Patient_01'
else:
    filepath = "../nii_data/"



class PG:
    
    def __init__(self) -> None:
        self.Patient = ""
        self.GT = ""
        self.Name = ""
        
    def Print(self) -> None:
        print(self.Patient)
        print(self.GT)
        
filenames = filepath + "Patient_02" 

_PG = PG()
for index,name in  enumerate(os.listdir(filenames)):
    # print( filepath + i ) 
    if name.startswith('Patient') and name.endswith('nii'):
        _PG.Patient = name
        _PG.Name = _PG.Patient.split(".")[0]
        print(_PG.Name)
        print(_PG.Patient)
        img = nib.load("/home/w0x7ce/Desktop/AQ/nii_data" + "/" + _PG.Name + "/" + _PG.Patient)  # 读取nii
        data = img.get_fdata()
        print(data.shape,type(data))
        
        slice_2 = data[:, :, 0]
        # print(slice_2)
        cv2.imwrite('./save_Patient.png',slice_2)
        
        # img = Image.open(slice_2)
        # img.save('{}_mask.png'.format(index))
        

    if name.startswith('GT') and name.endswith('nii'):
        _PG.GT = name
        
        img = nib.load("/home/w0x7ce/Desktop/AQ/nii_data" + "/" + _PG.Name + "/" + _PG.GT)
        data = img.get_fdata()
        print(data.shape,type(data))
        slice_2 = data[:, :, 0]
        cv2.imwrite('./save_GT.png',slice_2)
        print(_PG.GT)
        

    print(str(index) + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    
    
    
    # img = nib.load("/home/w0x7ce/Desktop/AQ/nii_data" + "/" + _PG.Name + "/" + _PG.Patient)  # 读取nii
    # break
    # img_fdata = img.get_fdata()
    # print(img_fdata)
    # fname = img_fdata.replace('.nii', '')  # 去掉nii的后缀名
    
    # (x, y, z) = img.shape 
    
    # if '.gz' in _PG.GT:
    #     (x, y, z, _h) = img.shape
    #     print("img2:")
    #     print(img.shape)
    # else:
    #     (x, y, z) = img.shape
    #     print("img3:")
    #     print(img.shape)
    
    
    # _PG.Print() 
    # tail = nii
    # if 


# exit
#
# for f in filenames:  # 开始读取nii文件
#     s = f[-4:]
#     print(s)
#
#     if s != '.nii':
#         continue
#     s1 = f[:-4]
#     print(s1)
#     imgfile_path = imgfile + s1
#     print("imgfile_path:" + imgfile_path)
#     img_path = os.path.join(filepath, f)

###
#     img = nib.load(img_path)  # 读取nii
#     print("img:")
#     print(img)
#     img_fdata = img.get_fdata()
#
#     fname = f.replace('.nii', '')  # 去掉nii的后缀名
#     img_f_path = os.path.join(imgfile, fname)
#     if not os.path.exists(img_f_path):
#         os.mkdir(img_f_path)
#
#     # 创建nii对应的图像的文件夹
#     # # if not os.path.exists(img_f_path):
#     # os.mkdir(img_f_path) #新建文件夹
#     # #开始转换为图像
#     if '.gz' in s1:
#         (x, y, z, _) = img.shape
#         print("img2:")
#         print(img.shape)
#     else:
#         (x, y, z) = img.shape
#         print("img3:")
#         print(img.shape)
#
#     for i in range(z):  # z是图像的序列
#         silce = img_fdata[:, :, i]  # 选择哪个方向的切片都可以
#         imageio.imwrite(os.path.join(img_f_path, '{}_mask.png'.format(i)), silce)
#         img = Image.open(os.path.join(img_f_path, '{}_mask.png'.format(i)))
#         img.save(os.path.join(img_f_path, '{}_mask.png'.format(i)))
# 