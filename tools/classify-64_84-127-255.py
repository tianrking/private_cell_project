import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import cv2
import shutil

class PG:
    
    def __init__(self) -> None:
        self.Patient = ""
        self.GT = ""
        self.Name = ""
        
    def Print(self) -> None:
        print(self.Patient)
        print(self.GT)

filenames = "/home/w0x7ce/Desktop/AP/label"
_dir = "/home/w0x7ce/Desktop/AP/"

class G:
        
        def __init__(self) -> None:
            self._255_127_85_64 = 0
            self._255_127_85 = 0
            self._255_127 = 0
            self._255 = 0
            self._0 =0
            self._maybe_error = []
            
        def analysis(self) -> int:
            
            return  self._255_127_85_64,self._255_127_85,self._255_127,self._255,self._0
        
        def get_error(self) -> str:
            
            return self._maybe_error
            
if not os.path.exists("CLASSIFY"):
    
    os.mkdir("CLASSIFY")
    
    os.mkdir("CLASSIFY/4")
    os.mkdir("CLASSIFY/3")
    os.mkdir("CLASSIFY/2")
    os.mkdir("CLASSIFY/1")
    os.mkdir("CLASSIFY/0")
    
    os.mkdir("CLASSIFY/4/image")
    os.mkdir("CLASSIFY/3/image")
    os.mkdir("CLASSIFY/2/image")
    os.mkdir("CLASSIFY/1/image")
    os.mkdir("CLASSIFY/0/image")
    
    os.mkdir("CLASSIFY/4/label")
    os.mkdir("CLASSIFY/3/label")
    os.mkdir("CLASSIFY/2/label")
    os.mkdir("CLASSIFY/1/label")
    os.mkdir("CLASSIFY/0/label")

for index,name in  enumerate(os.listdir(filenames)):

    _dir = "/home/w0x7ce/Desktop/AP/"
    _dir_label = _dir + "label/" + name
    _dir_image = _dir + "images/" + name
    
    bb = G()
    
    temp = cv2.imread(_dir_label,cv2.IMREAD_GRAYSCALE)
    
    if temp is None :
        bb._maybe_error.append(temp)
        continue
    
    print(_dir_label)

    _64_lock = 1
    _85_lock = 1
    _127_lock = 1
    _255_lock = 1

    for uu in temp:

        for ii in uu:
                    
            if ii == 64 and _64_lock:
                print(ii)
                _64_lock = 0
            
            if ii == 85 and _85_lock:
                print(ii)
                _85_lock = 0
                
            if ii == 127 and _127_lock:
                print(ii)
                _127_lock = 0
                
            if ii == 255 and _255_lock:
                print(ii)
                _255_lock = 0

    #  5 -> 4+3+2+1 = 10
    
    # ok 1
    if not _64_lock and not _127_lock and not _255_lock :
        bb._255_127_85_64 = bb._255_127_85_64 + 1
        shutil.copy(_dir_label, "CLASSIFY/4/image")
        shutil.copy(_dir_image, "CLASSIFY/4/label")
        print("DDDDDDDDDDDDD")
    
    # ok  
    if  _64_lock and not _85_lock and _127_lock and not _255_lock :
        bb._255_127_85 = bb._255_127_85 + 1
        shutil.copy(_dir_label, "CLASSIFY/2/image")
        shutil.copy(_dir_image, "CLASSIFY/2/label")
        print("BBBBBBBBBBBBBB")
        
    # ok 
       
    if  _64_lock and _85_lock and _127_lock and not _255_lock :
        bb._255_127 = bb._255_127 + 1
        shutil.copy(_dir_label, "CLASSIFY/1/label")
        shutil.copy(_dir_image, "CLASSIFY/1/image")
        print("cccccccccccccc")
    
    # ok 
       
    if  _64_lock and _85_lock and _127_lock and _255_lock :
        bb._255_127 = bb._255_127 + 1
        shutil.copy(_dir_label, "CLASSIFY/3/label")
        shutil.copy(_dir_image, "CLASSIFY/3/image")
        print("EEEEEEEEEEEEEE")
      
    # ok 
       
    if  _64_lock and not _85_lock and _127_lock and _255_lock :
        bb._255_127 = bb._255_127 + 1
        shutil.copy(_dir_label, "CLASSIFY/2/label")
        shutil.copy(_dir_image, "CLASSIFY/2/image")
        print("EEEEEEEEEEEEEE")
    

print(bb.analysis())
print(bb.get_error())

# Fuck 85
# kk = "label/Patient04_27_mask.png"
# temp = cv2.imread(kk,cv2.IMREAD_GRAYSCALE)

# import numpy as np
# o = []
# for ii in temp:
#     for uu in ii:
#         o.append(uu)

# o = np.unique(o)
# print(o)
