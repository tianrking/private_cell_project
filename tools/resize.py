#todo 

from PIL import Image
import os

class class_Resize_dir:
    
    def __init__(self,_origin,_resize):
        self.origin = _origin
        self.resize = _resize

class class_Resize:
    
    def __init__(self,_dir,_name):
        
        self.root = _dir # without /
        self.pic_origin = Image.open(self.root + "/" + _name)
        self.pic_resize = None
        
        self.resize_x = 0
        self.resize_y = 0
        
    def resize(self,x,y):
        
        self.resize_x , self.resize_y = x,y
        self.pic_resize = self.pic_origin.resize((self.resize_x,self.resize_y))
    
    def resize_save(self,_dir,_name):
        # without /
        self.pic_resize.save(_dir + "/" + _name, ignore_discard=False, ignore_expires=False)


# dic_name = "/home/w0x7ce/Desktop/private_cell_project/media"

dir_name = r"F:\ww\lwd\data_only\Data\yuanhe_heren\\test"

_dir = class_Resize_dir(dir_name +  "\label-clip",dir_name + "\label-clip-resize")

dic_name_list = os.walk(_dir.origin)

t = 1
for path, dir_list, file_list in dic_name_list:
    for file_name in file_list:
        
        dd = class_Resize(path,file_name)
        dd.resize(200,200)
        dd.resize_save(_dir.resize,file_name)
        dd = 0
        
        print(str(t) + " " +path + "/" + file_name )
        t = t + 1
        # print(_dir.origin + "/" + file_name)
        # print(_dir.resize + "/" + file_name)
