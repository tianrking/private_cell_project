import threading
import time

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import os

import sys,platform

import pandas as pd

class _covert_image1:
    def __init__(self) -> None:
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0

        self.x_list = []
        self.y_list = []

        pass

    def read_image(self, input_dir) -> None:

        self.im_origin = Image.open(input_dir)
        self.im_np = np.array(self.im_origin)
        self.im_np_mask = self.im_np.copy()
        self.image_convert = ""

    def read_label(self, input_dir) -> None:

        self.lb_origin = Image.open(input_dir)
        self.lb_np = np.array(self.lb_origin)

        self.image_convert = ""

    def set_color(self, color) -> None:
        self.color = color  # R+G+B

    def get_mask(self) -> Tuple[int, int, int, int]:

        for x in range(0, 500):
            for y in range(0, 500):
                if self.lb_np[x][y].sum() == self.color:
                    self.x_list.append(x)
                    self.y_list.append(y)
                #   continue
                # self.im_np_mask[x][y] = np.array([0,0,0]) #wrong

        self.x_min = min(self.x_list)
        self.x_max = max(self.x_list)
        self.y_min = min(self.y_list)
        self.y_max = max(self.y_list)

        for x in range(0, 500):
            for y in range(0, 500):
                if x > self.x_min and x < self.x_max:
                    if y > self.y_min and y < self.y_max:
                        continue
                self.im_np_mask[x][y] = np.array([0, 0, 0])

        self.x_list = []
        self.y_list = []

    def find_edge(self) -> None:
        print(self.x_min, self.y_min)
        print(self.x_max, self.y_max)

    def get_edge(self) -> Tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def display_im(self) -> None:
        plt.imshow(self.im_np[self.x_min:self.x_max, self.y_min:self.y_max])
        plt.show()

    def display_lb(self) -> None:
        plt.imshow(self.lb_np[self.x_min:self.x_max, self.y_min:self.y_max])
        plt.show()

    def display_origin(self) -> None:
        plt.subplot(1, 2, 1)
        plt.imshow(self.im_np)
        plt.subplot(1, 2, 2)
        plt.imshow(self.lb_np)
        plt.show()

    def display_result(self) -> None:
        plt.subplot(1, 2, 1)
        plt.imshow(self.lb_np[self.x_min:self.x_max, self.y_min:self.y_max])
        plt.subplot(1, 2, 2)
        plt.imshow(self.im_np[self.x_min:self.x_max, self.y_min:self.y_max])
        plt.show()

    def display_mask(self) -> None:
        plt.subplot(1, 2, 1)
        plt.imshow(self.im_np_mask)
        plt.subplot(1, 2, 2)
        plt.imshow(self.lb_np)
        plt.show()

    def add_mask(self) -> None:
        pass

    def save_mask(self, save_dir) -> None:

        im = Image.fromarray(self.im_np_mask)
        im.save(save_dir)
        im = []

    def test_api(self) -> None:
        x, y = 1, 1
        print(self.lb_np[x][y])
        # self.lb_np.setflags(write=1)
        # self.im_np.flags.writeable = True
        # self.lb_np[x][y] = np.array([1,2,3])
        # print(type(self.lb_np[x][y]))


# heren_yuanhe_dir = "/content/cell_data/heren-yuanhe/"
# heren_yuanhe_image_dir = heren_yuanhe_dir + "/image_all/"
# heren_yuanhe_label_dir = heren_yuanhe_dir + "/label_all/"
# heren_yuanhe_image_roi_dir = heren_yuanhe_dir + "/image_roi_all/"


# heren_yuanhe_dir = "/content/cell_data/heren-yuanhe-zhiyun-xibao"
# heren_yuanhe_image_dir = heren_yuanhe_dir + "/image_all/"
# heren_yuanhe_label_dir = heren_yuanhe_dir + "/label_all/"
# heren_yuanhe_image_roi_dir = heren_yuanhe_dir + "/image_roi_all/"


# if platform.system == "Windows":
#     heren_yuanhe_dir = r"E:\w0x7ce_td\A\heren-yuanhe-zhiyun-xibao"
#     heren_yuanhe_image_dir = heren_yuanhe_dir + r"\image_all\\"
#     heren_yuanhe_label_dir = heren_yuanhe_dir + r"\label_all\\"
#     heren_yuanhe_image_roi_dir = heren_yuanhe_dir + r"\image_roi_all\\"
# else: # platform.system == "Linux
#     heren_yuanhe_dir = "/content/cell_data/heren-yuanhe/"
#     heren_yuanhe_image_dir = heren_yuanhe_dir + "/image_all/"
#     heren_yuanhe_label_dir = heren_yuanhe_dir + "/label_all/"
#     heren_yuanhe_image_roi_dir = heren_yuanhe_dir + "/image_roi_all/"

heren_yuanhe_dir = r"E:\w0x7ce_td\A\heren-yunahe-zhiyun"
heren_yuanhe_image_dir = heren_yuanhe_dir + r"\image_all\\"
heren_yuanhe_label_dir = heren_yuanhe_dir + r"\label_all\\"
heren_yuanhe_image_roi_dir = heren_yuanhe_dir + r"\image_roi_all\\"

if os.path.exists(heren_yuanhe_image_roi_dir) == 0:
    os.mkdir(heren_yuanhe_image_roi_dir)
    print("Create floder")
else:
    print("DAN")
    
time = 1
df = pd.DataFrame(columns=('index','error'))

for root, dirs, files in os.walk(heren_yuanhe_image_dir):
    print(root,dirs,files)
    for file in files:

        file_name = file[0:-4]
        
        
        print(file_name)
        print(time, file_name)
        
        
        time = time + 1
        AA = _covert_image1()

        try:
            print(heren_yuanhe_image_dir + file_name + ".JPG")
            AA.read_image(heren_yuanhe_image_dir + file_name + ".JPG")
        except:
            print(heren_yuanhe_image_dir + file_name + ".png")
            AA.read_image(heren_yuanhe_image_dir + file_name + ".png")
        try:
            AA.read_label(heren_yuanhe_label_dir + file_name + ".JPG")
        except:
            AA.read_label(heren_yuanhe_label_dir + file_name + ".png")

        try:
            AA.set_color(192*3)  # 255 * 3 ## 192 * 3 
            AA.get_mask()
            AA.save_mask(heren_yuanhe_image_roi_dir + file_name + ".png")
        except:
            df.append([{'error':file_name}],ignore_index=True)
            
        # AA.display_origin()
        # AA.display_result()
        # AA.display_mask()

        
        # x0, y0, x1, y1 = AA.get_edge()

# AA = _covert_image1()
# AA.read_image(heren_yuanhe_image_list[0])
# AA.read_label(heren_yuanhe_label_list[0])
# AA.set_color(255*3)
# AA.get_mask()
# AA.display_origin()
# AA.display_result()
# AA.display_mask()

# AA.save_mask(heren_yuanhe_image_roi_dir+"a.jpg")
# x0,y0,x1,y1 = AA.get_edge()
# AA.test_api()
# print("test")