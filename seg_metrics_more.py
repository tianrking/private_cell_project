import numpy as np
import cv2
import os
import xlwt
""" 
混淆矩阵
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""
#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if(len(colorDict) == classNum):
            break
    #  存储颜色的BGR字典，用于预测时的渲染结果
    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位B,中3位G,后3位R
        color_BGR = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_BGR.append(color_BGR)
    #  转为numpy格式
    colorDict_BGR = np.array(colorDict_BGR)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1 ,colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY

def ConfusionMatrix(numClass, imgPredict, Label):  
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def Precision(confusionMatrix):  
    #  返回所有类别的精确率precision  
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return precision  

def Recall(confusionMatrix):
    #  返回所有类别的召回率recall
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score
def IntersectionOverUnion(confusionMatrix):  
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU
def dicesimilaritycoefficient(confusionMatrix):
    intersection = np.diag(confusionMatrix)
    sum = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0)
    Dice = (2*intersection)/sum
    return Dice

def MeanIntersectionOverUnion(confusionMatrix):  
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

#################################################################


excel = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = excel.add_sheet('test', cell_overwrite_ok=True)
namelist = ["胚胎序号", "指标", "背景", "原核", "质晕", "细胞", "均值"]
for num in range(len(namelist)):
    sheet.write(0, num, namelist[num])
number = 1
image_filepath = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\multi_predict\double_unet"
label_filepath = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\label_file"
filelist = os.listdir(image_filepath)
for filename in filelist:
    #  标签图像文件夹
    LabelPath = label_filepath + '/' + filename
    #  预测图像文件夹
    PredictPath = image_filepath + '/' + filename
    #  类别数目(包括背景)
    classNum = 4
    #################################################################

    #  获取类别颜色字典
    colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

    #  获取文件夹内所有图像
    labelList = os.listdir(LabelPath)
    PredictList = os.listdir(PredictPath)

    #  读取第一个图像，后面要用到它的shape
    Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)

    #  图像数目
    label_num = len(labelList)

    #  把所有图像放在一个数组里
    label_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
    predict_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
    for i in range(label_num):
        Label = cv2.imread(LabelPath + "//" + labelList[i])
        Label = cv2.cvtColor(Label, cv2.COLOR_BGR2GRAY)
        label_all[i] = Label
        Predict = cv2.imread(PredictPath + "//" + PredictList[i])
        Predict = cv2.cvtColor(Predict, cv2.COLOR_BGR2GRAY)
        predict_all[i] = Predict

    #  把颜色映射为0,1,2,3...
    for i in range(colorDict_GRAY.shape[0]):
        label_all[label_all == colorDict_GRAY[i][0]] = i
        predict_all[predict_all == colorDict_GRAY[i][0]] = i

    #  拉直成一维
    label_all = label_all.flatten()
    predict_all = predict_all.flatten()

    #  计算混淆矩阵及各精度参数
    confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
    precision = Precision(confusionMatrix)
    recall = Recall(confusionMatrix)
    OA = OverallAccuracy(confusionMatrix)
    IoU = IntersectionOverUnion(confusionMatrix)
    Dice = dicesimilaritycoefficient(confusionMatrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
    mIOU = MeanIntersectionOverUnion(confusionMatrix)
    f1ccore = F1Score(confusionMatrix)

    col = 0
    sheet.write(number, col, filename)
    col = col + 1
    sheet.write(number, col, "Precision")
    col = col + 1
    for k in range(4):
        sheet.write(number, col, precision[k])
        col = col + 1
    sheet.write(number, col, np.mean(precision))
    number = number + 1

    col = 1
    sheet.write(number, col, "Recall")
    col = col + 1
    for k in range(4):
        sheet.write(number, col, recall[k])
        col = col + 1
    sheet.write(number, col, np.mean(recall))
    number = number + 1

    col = 1
    sheet.write(number, col, "IoU")
    col = col + 1
    for k in range(4):
        sheet.write(number, col, IoU[k])
        col = col + 1
    sheet.write(number, col, mIOU)
    number = number + 1

    col = 1
    sheet.write(number, col, "Dice")
    col = col + 1
    for k in range(4):
        sheet.write(number, col, Dice[k])
        col = col + 1
    sheet.write(number, col, np.mean(Dice))
    number = number + 1

excel.save(r'doubleunet评价指标.xls')