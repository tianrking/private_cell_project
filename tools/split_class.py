import cv2
from PIL import Image

class FIND_LOCATION:

  x_min = 0
  x_max = 0
  y_min = 0
  y_max = 0
  
  x_list = []
  y_list = []

  np_im = 0

  ROI = 0

  def __init__(self,dir):
    im = Image.open(dir)
    self.np_im = np.asarray(im)

  def read_img(self,dir):
    im = Image.open(dir)
    self.np_im = np.asarray(im)
    return self.np_im

  def find_white(self):

    for x in range(0,500):
      for y in range(0,500):
        if(self.np_im[x][y].sum()==765):
          self.x_list.append(x)
          self.y_list.append(y)
    
    self.x_min = min(x_list)
    self.x_max = max(x_list)
    self.y_min = min(y_list)
    self.y_max = max(y_list)

  def get_location(self):
    return self.y_min,self.x_min, self.y_max, self.x_max

  def clip_img(self,clip_origin_location,save_location):
    gg = cv2.imread(clip_origin_location)
    gg_copy = gg.copy()
    self.ROI = gg_copy[x_min:x_max,y_min:y_max]
    cv2.imwrite(save_location,self.ROI)
