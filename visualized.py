import numpy as np

color_dict = {"black": [0, 0, 0],
              "red": [238, 0, 0], 
              "pink": [255, 192, 203],
			  "yellow":[255,255,0],
			  "green":[0,255,0],
			  "blue":[0,0,255]}

def gray2rgb(gray, color_dict):
    """
    convert gray image into RGB image
    :param gray: single channel image with numpy type
    :param color_dict: color map
    :return:  rgb image
    """
    # 1：创建新图像容器
    rgb_image = np.zeros(shape=(*gray.shape, 3))
    # 2： 遍历每个像素点
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            # 3：对不同的灰度值选择不同的颜色
            if gray[i, j] < 1:
                rgb_image[i, j, :] = color_dict["black"]
            elif 2 >= gray[i, j] >= 1:
                rgb_image[i, j, :] = color_dict["red"]
            elif 3 >= gray[i, j] >= 2:
                rgb_image[i, j, :] = color_dict["green"]
            elif 4 >= gray[i, j] >= 3:
                rgb_image[i, j, :] = color_dict["blue"]
            else:
                rgb_image[i, j, :] = color_dict["pink"]

    return rgb_image.astype(np.uint8)

from PIL import Image
import matplotlib.pyplot as plt
img=np.array(Image.open(r'F:\ww\project\dataset_formal\final_data\E0001\D2021.01.20_S00120_I0939_D_WELL01_RUN212.png').convert('L'))

plt.figure("lena")
arr=img.flatten()
n, bins, patches = plt.hist(arr, bins=130, density=0, facecolor='green', alpha=0.75)
plt.show()


