import os
import cv2

'''
读取文件夹下图片并保存到指定路径  
'''
readpathname = r"F:\ww\lwd\Unet_RSimage_Multi-band_Multi-class-master\Data\train_yuanhe\image_goule_file\E0013"
outpathname = r"F:\ww\lwd\Unet_RSimage_Multi-band_Multi-class-master\Data\train_yuanhe\label_file\E0013"
def read_path(file_pathname, save_pathname):
	#遍历该目录下的所有图片文件
	for filename in os.listdir(file_pathname):
		# print(filename)
		img = cv2.imread(file_pathname+'/'+filename)
		# print(img[0, 0])
		####change to gray
	  #（下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）不需要这种操作只需注释掉即可
		# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# image_np=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
		for i in range(500):
			green = 0
			green0 = 0
			g_count = 0
			blue = 0
			blue0 = 0
			b_count = 0
			# p_count = 0
			for j in range(500):
				# 找绿色
				# sc1 = int(img[i, j][1]) - int(img[i, j][0]) # G-B
				# sc2 = int(img[i, j][1]) - int(img[i, j][2]) # G-R
				# if img[i, j][0] >= 0 and img[i, j][1] >= 90 and img[i, j][2] > 0 and sc1 > 30 and sc2 > 30:

				# 找绿色和蓝色
				sc1 = int(img[i, j][1]) - int(img[i, j][0]) # G-B # 绿色
				sc2 = int(img[i, j][1]) - int(img[i, j][2]) # G-R # 绿色
				sc3 = int(img[i, j][0]) - int(img[i, j][1]) # B-G # 蓝色
				sc4 = int(img[i, j][0]) - int(img[i, j][2]) # B-R # 蓝色
				# sc5 = int(img[i, j][0]) - int(img[i, j][1]) # B-G # 紫色
				# sc6 = int(img[i, j][2]) - int(img[i, j][1]) # R-G # 紫色
				sc7 = int(img[i, j][2]) - int(img[i, j][1]) # R-G # 红色
				sc8 = int(img[i, j][2]) - int(img[i, j][0]) # R-B # 红色
				if img[i, j][0] >= 0 and img[i, j][1] >= 90 and img[i, j][2] > 0 and sc1 > 30 and sc2 > 30:# 找绿色
					green = j
					img[i,j] = [1, 1, 1]
					if g_count == 0:
						img[i, j] = (255, 255, 255)
						green0 = j
						g_count= g_count + 1 # 目的为了每行只找两个点
				# elif img[i, j][0] >= 90 and img[i, j][1] >= 0 and img[i, j][2] >= 0 and sc3 > 30 and sc4 > 30:# 找蓝色
				# 	blue = j
				# 	img[i,j] = [1, 1, 1]
				# 	if b_count == 0:
				# 		img[i, j] = (127, 127, 127)
				# 		blue0 = j
				# 		b_count = b_count + 1
				# elif img[i, j][0] >= 90 and img[i, j][1] >= 0 and img[i, j][2] >= 90 and sc5 > 30 and sc6 > 30:# 找紫色
				# 	if p_count == 0:
				# 		img[i, j] = (63, 63, 63)
				# 		p_count = p_count + 1
				elif img[i, j][0] >= 0 and img[i, j][1] >= 0 and img[i, j][2] >= 90 and sc7 > 30 and sc8 > 30:# 找红色
					blue = j
					img[i,j] = [1, 1, 1]
					if b_count == 0:
						img[i, j] = (127, 127, 127)
						blue0 = j
						b_count = b_count + 1
				else:
					img[i, j] = (0, 0, 0)
				if green - green0 > 4:
					img[i, green] = (255, 255, 255)
				if blue - blue0 > 4:
					img[i, blue] = (127, 127, 127)

		# cv2.imwrite(r"F:\ww\lwd\Unet_RSimage_Multi-band_Multi-class-master\Data\train_yuanhe\img" + "/" + filename[:-4] + ".png", img)
		img0 = img.copy()
		# print(img[0, 0])
		# for循环做出蓝色标记的区域生成结果图
		for i in range(img.shape[0]):
			rightpoint127 = 0
			leftpoint127 = img.shape[1]
			green1 = 0
			green2 = 0
			# purple = 0
			count = 0
			for j in range(img.shape[1]):
				if all(img[i, j] == (127, 127, 127)):
					if j < leftpoint127:
						leftpoint127 = j
					if j > rightpoint127:
						rightpoint127 = j
				elif all(img[i, j] == (255, 255, 255)):
					if count == 0:
						green1 = j
						count = count + 1
					else:
						green2 = j
			# print("shangshang")
			# print(leftpoint127, rightpoint127)
			if leftpoint127 == rightpoint127:
				if green1 < leftpoint127 and green2 < leftpoint127:
					leftpoint127 = green2
				elif green1 > leftpoint127 and green2 > leftpoint127:
					rightpoint127 = green1
				# print(leftpoint127, rightpoint127)
			# print("shang")
			# print(leftpoint127, rightpoint127)
			for s in range(leftpoint127, rightpoint127 + 1):
				img[i, s] = (255, 255, 255)

			# kernel = np.ones((3, 3), np.uint8)
			# thresh = cv2.dilate(thresh, kernel, iterations=1)
			# thresh = cv2.erode(thresh, kernel, iterations=1)
			# thresh = cv2.erode(thresh, kernel, iterations=1)
			# thresh = cv2.dilate(thresh, kernel, iterations=1)
		out1= cv2.medianBlur(img, ksize=9)
		# cv2.imwrite(r"F:\ww\lwd\Unet_RSimage_Multi-band_Multi-class-master\Data\train_yuanhe\out1" + "/" + filename[:-4] + ".png", out1)
		# for循环做出绿色标记的结果图
		for i in range(img0.shape[0]):
			rightpoint255 = 0
			leftpoint255 = img0.shape[1]
			blue1 = 0
			blue2 = 0
			# purple = 0
			count = 0
			for j in range(img0.shape[1]):
				# print(img0[i, j])
				if all(img0[i, j] == (255, 255, 255)):
					# print("in")
					if j < leftpoint255:
						leftpoint255 = j
					if j > rightpoint255:
						rightpoint255 = j
				elif all(img0[i, j] == (127, 127, 127)):
					if count == 0:
						blue1 = j
						count = count + 1
					else:
						blue2 = j


			if leftpoint255 == rightpoint255:
				if blue1 < leftpoint255 and blue2 < leftpoint255:
					leftpoint255 = blue2
				elif blue1 > leftpoint255 and blue2 > leftpoint255:
					rightpoint255 = blue1
			# print(leftpoint255, rightpoint255)
			for s in range(leftpoint255, rightpoint255 + 1):
				img0[i, s] = (255, 255, 255)
			# out2 = cv2.dilate(out, kernel, iterations=1)
		out2 = cv2.medianBlur(img0, ksize=9)
		# cv2.imwrite(r"F:\ww\lwd\Unet_RSimage_Multi-band_Multi-class-master\Data\train_yuanhe\out2" + "/" + filename[:-4] + ".png", out2)
		# for循环找出蓝色的结果（第一个for循环的结果）中为（255， 255， 255）的像素点位置，
		# 在绿色图中（第二个for循环结果图）渲染为（255,255,255）
		for  i in range(out1.shape[0]):
			for j in range(out1.shape[1]):
				if all(out1[i, j] == (255, 255, 255)):
					out2[i, j] = (255, 255, 255)
		ret, result= cv2.threshold(out2, 128, 255, cv2.THRESH_BINARY)
		# #####save figure
		cv2.imwrite(save_pathname+"/"+filename[:-4] + ".png", result) #只能是png
		# cv2.imwrite(outpathname+"/"+filename, img)
#注意*处如果包含家目录（home）不能写成~符号代替
#读取的目录
for i in range(1):
	if i+1 < 10:
		newreadpath = readpathname.replace(readpathname[-1], str(i+1))
		newoutpath = outpathname.replace(outpathname[-1], str(i+1))
	if i+13 >= 10:
		newreadpath = readpathname.replace(readpathname[-2:], str(i+13))
		newoutpath = outpathname.replace(outpathname[-2:], str(i+13))
	# print(newpath)
	read_path(newreadpath, newoutpath)
