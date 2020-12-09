import sys
import json
import math
import numpy as np
import cv2
import math
from PIL import Image
import os
import itertools
import random

# NAMRTILIES=['C','Sun','Ye','L']

#shap原图尺寸、canvas2遮挡实例所在的图（裁剪图）、canvas1要遮挡的图、shap原图尺寸、mask 0，1掩码、imgCid类别掩码

cropthresholdmin = 0.4
cropthresholdmax = 1
iouthresholdmax = 0.45
iouthresholdmin = 0.1
people_enhance_number = 40  # 要增加多少张这类增强图
max_add_peoples_mumber = 1  # 一张图最多加多少个遮挡


# 随机生成用来遮档的实例的bbox在要遮挡图上的左上角位置（随机遮挡）
def cal_xy(shape):
    xnew = random.randint(0, shape[1])
    ynew = random.randint(0, shape[0])
    newxy = (xnew, ynew)
    return newxy


# bbox超过要遮挡图的一个范围，所以做一个裁剪
def cal_people_bbox(canvas2, mask2, imgCid2):
    findresult = np.where(mask2 != 0)
    xmin, ymin, xmax, ymax, w, h = 0, 0, 0, 0, 0, 0
    if findresult[0].size > 0:
        ymin = int(findresult[0].min())
        ymax = int(findresult[0].max())
        xmin = int(findresult[1].min())
        xmax = int(findresult[1].max())
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    bbox = [xmin, ymin, w, h]
    canvas2 = canvas2[ymin:ymin + h, xmin:xmin + w]
    mask2 = mask2[ymin:ymin + h, xmin:xmin + w]
    imgCid2 = imgCid2[ymin:ymin + h, xmin:xmin + w]
    return canvas2, mask2, imgCid2

#计算原图（用来遮挡实例的）中的原始xymin和放到要遮挡图中的新xymin之间的差值
def cal_deltaxy(imgHid2, person2, xymin):
    findresult = np.where(imgHid2 == person2)
    xmin_in_2, ymin_in_2 = 0, 0
    if findresult[0].size > 0:
        ymin_in_2 = int(findresult[0].min())
        xmin_in_2 = int(findresult[1].min())
    deltax = xmin_in_2 - xymin[0]
    deltay = ymin_in_2 - xymin[1]
    return deltax, deltay

#需要经过裁剪的实例，修改后保存下的内容（shap、mask、imgCid）
def cal_people_crop(shape, canvas2, xymin, mask2, imgCid2):
    cropscale = 0
    w2 = canvas2.shape[1]
    h2 = canvas2.shape[0]
    deltaw = max(0, xymin[0] + w2 - shape[1])
    deltah = max(0, xymin[1] + h2 - shape[0])
    # cropscale=float((w2-deltaw)*(h2-deltah))/float(w2*h2)
    canvas2 = canvas2[0:h2 - deltah, 0:w2 - deltaw]
    fenmu = np.sum(mask2 == 1)
    mask2 = mask2[0:h2 - deltah, 0:w2 - deltaw]
    fenzi = np.sum(mask2 == 1)
    cropscale = float(fenzi) / fenmu
    imgCid2 = imgCid2[0:h2 - deltah, 0:w2 - deltaw]
    return cropscale, canvas2, mask2, imgCid2


def readimagename(rootpath, TYPE):
    path = rootpath + TYPE + '.txt'
    allfile = np.loadtxt(path, delimiter=',')
    fileindexs = []
    imagenames = []
    for i in allfile:
        inputid = int(i)
        fileindexs.append(inputid)
        name = "%07d" % inputid
        imagefile = rootpath + 'Images/' + name + '.jpg'
        imagenames.append(imagefile)
    return fileindexs, imagenames


def cal_people_maskiou(imgHid, canvas2, mask2, xymin, iouthresholdmax, iouthresholdmin):
    instances_number = imgHid.max()
    instances_number2 = (np.unique(imgHid)).size - 1 #np.unique() 去除数组中的重复数字，并进行排序之后输出
    if instances_number != instances_number2:
        return False
    # alliou = 0
    maxiou = 0
    validmask = 0
    #记录下遮挡的实例与图中的实例之间的IoU，求最大的实例遮挡情况下的IoU不能超过阈值，并且也不能过小
    for instance_index in range(1, instances_number + 1):
        mask1 = (imgHid == instance_index)
        iou = people_maskiou(mask1, mask2, xymin)
        if iou > 0:
            validmask = validmask + 1
            maxiou = max(maxiou, iou)
            # alliou = alliou + iou
            print('iou:',iou)
            print('maxiou:', maxiou)
            if maxiou > iouthresholdmax:
                return False

    if validmask > 0:
        # avgiou = alliou / validmask
        # print('avgiou:',avgiou)
        if maxiou > iouthresholdmax or maxiou < iouthresholdmin:
            return False

    if validmask == 0:
        return False
    return True


def people_maskiou(mask1, mask2, xymin):
    jiao = 0
    bing1 = np.where(mask1 == 1)[0].size
    bing2 = np.where(mask2 == 1)[0].size
    iou = 0
    for i in range(0, mask2.shape[0]):
        for j in range(0, mask2.shape[1]):
            if mask1[xymin[1] + i][xymin[0] + j] == 1 and mask2[i][j] == 1:
                jiao = jiao + 1
    # if bing1+bing2-jiao:
    #    iou=jiao/(bing1+bing2-jiao)
    if bing1:
        iou = jiao / bing1
    return iou


#将遮挡的实例添加到要被遮挡的图片中
def people_draw(imgHid, imgCid, imgIid, jsondata1, canvas1, xymin, canvas2, mask2, imgCid2):
    instance_number = imgHid.max()
    newperson = instance_number + 1
    for i in range(0, mask2.shape[0]):
        for j in range(0, mask2.shape[1]):
            if mask2[i][j] == 1:
                imgHid[i + xymin[1]][j + xymin[0]] = newperson
                imgCid[i + xymin[1]][j + xymin[0]] = imgCid2[i][j]
                imgIid[i + xymin[1]][j + xymin[0]] = (newperson - 1) * 20 + imgCid2[i][j]
                canvas1[i + xymin[1]][j + xymin[0]] = canvas2[i][j]

    # jsondata1 = updatajson(jsondata1, imgHid, imgCid, imgIid)
    return imgHid, imgCid, imgIid, jsondata1, canvas1



# def updatajson(dictwrite, newHid, newCid, newIid):
#     instance_number, resolution, partlabels = myjson.writejson_2(newHid, newCid, newIid)
#     dictwrite['Instance_number'] = int(instance_number)
#     # print('dictwrite_Instance_number:',dictwrite['Instance_number'])
#     dictwrite['Resolution'] = resolution
#     dictwrite['Parts_number'] = int(partlabels.size)
#     dictwrite['Parts'] = partlabels.tolist()
#     dictwrite = myjson.writejson_instances(dictwrite, newHid, newCid, resolution)
#     # print(dictwrite)
#     # dictwrite=myjson.writejson_segiou(dictwrite,newHid)
#     dictwrite = myjson.writejson_segiou(dictwrite, newHid)
#     return dictwrite

#保存json的相关信息
def writecrowd(newHid, newCid, newIid, newjsondata, newimg, writepath, fileindexnew, fileindexold):
    cv2.imwrite(writepath + 'Images/' + "%07d" % fileindexnew + '.jpg', newimg)
    newjsondata['Filename'] = "%07d" % fileindexnew
    newjsondata['Imagepath'] = writepath + 'Images/' + "%07d" % fileindexnew + '.jpg'
    newjsondata['Human_ids_path'] = writepath + 'Human_ids/' + "%07d" % fileindexnew + '.png'
    newjsondata['Category_ids_path'] = writepath + 'Category_ids/' + "%07d" % fileindexnew + '.png'
    newjsondata['Instance_ids_path'] = writepath + 'Instance_ids/' + "%07d" % fileindexnew + '.png'
    newjsondata['OutputImagepath'] = ""
    newjsondata['Humansimg_path'] = ""
    newjsondata['Categoriesimg_path'] = ""
    newjsondata['Instancesimg_path'] = ""
    newjsondata['DataAugmentation']['Is_Augmentation'] = {
        'Is_Augmentation': True,
        'Type': "Crowd_Occlusion_Augmentation",
        'Original_Image': fileindexold
    }
    image = Image.fromarray(newHid)
    image.save(newjsondata['Human_ids_path'])
    image = Image.fromarray(newCid)
    image.save(newjsondata['Category_ids_path'])
    image = Image.fromarray(newIid)
    image.save(newjsondata['Instance_ids_path'])
    writejsonfile(newjsondata, writepath + 'jsonfile/json' + "%07d" % fileindexnew + '.json')


def writejsonfile(dictwrite, outjsonfile):
    newjsondata = json.dumps(dictwrite, indent=4)
    with open(outjsonfile, 'w') as fw:
        fw.write(newjsondata)


#返回是否能遮挡以及遮挡的相关信息
def people(imgHid1, canvas2, mask2, imgCid2):
    shape = (imgHid1.shape[0], imgHid1.shape[1])
    xymin = cal_xy(shape)
    canvas2, mask2, imgCid2 = cal_people_bbox(canvas2, mask2, imgCid2)
    success = False
    #记录裁剪后的信息
    cropscale, canvas2, mask2, imgCid2 = cal_people_crop(shape, canvas2, xymin, mask2, imgCid2)
    # print('cropscale,canvas2.shape after crop:',cropscale,canvas2.shape)
    #保证用来遮挡的实例大于最小的阈值
    if cropscale > cropthresholdmin:
        success = cal_people_maskiou(imgHid1, canvas2, mask2, xymin, iouthresholdmax, iouthresholdmin)

    return success, canvas2, mask2, xymin, imgCid2


def readallinformation(rootpath, imagepath, fileindex):
    image = imagepath + "%07d" % fileindex + '.jpg'
    imgHid, imgCid, imgIid = readCIHP(fileindex, rootpath)
    jsondata = readjson(rootpath + 'jsonfile/json' + "%07d" % fileindex + '.json')
    canvas = cv2.imread(image)
    return imgHid, imgCid, imgIid, jsondata, canvas


def readjson(injsonfile):
    jsondata = None
    with open(injsonfile, 'rb') as f:
        jsondata = json.load(f)
    return jsondata


def readCIHP(fileindex, rootpath):
    name = "%07d" % fileindex
    imageHid = rootpath + 'Human_ids/' + name + '.png'
    imageCid = rootpath + 'Category_ids/' + name + '.png'
    imageIid = rootpath + 'Instance_ids/' + name + '.png'
    imgHid = np.array(Image.open(np.str(imageHid)))
    imgCid = np.array(Image.open(np.str(imageCid)))
    imgIid = np.array(Image.open(np.str(imageIid)))
    return imgHid, imgCid, imgIid


#一次遮挡增强的操作
def people_enhance_once(imgHid1, imgCid1, imgIid1, jsondata1, canvas1):
    sucess = False
    imgindex2 = random.randint(0, imagenumber - 1)
    fileindex2 = fileindexs[imgindex2]
    # fileindex2=random.randint(0,500)

    imgHid2, imgCid2, imgIid2, jsondata2, canvas2 = readallinformation(rootpath, imagepath, fileindex2)
    person2 = random.randint(1, jsondata2['Instance_number'])
    mask2 = (imgHid2 == person2)

    success, canvas2, mask2, xymin, imgCid2 = people(imgHid1, canvas2, mask2, imgCid2)
    # print('success,canvas2.shape,xymin:',success,canvas2.shape,xymin)
    deltax, deltay = 0, 0
    if success:
        imgHid1, imgCid1, imgIid1, jsondata1, canvas1 = people_draw(imgHid1, imgCid1, imgIid1, jsondata1, canvas1,
                                                                    xymin, canvas2, mask2, imgCid2)
        deltax, deltay = cal_deltaxy(imgHid2, person2, xymin)
    # print('success:',success)
    return success, imgHid1, imgCid1, imgIid1, jsondata1, canvas1, fileindex2, person2, deltax, deltay


#多次遮挡增强
def people_enhance(fileindex1, rootpath, imagepath):
    add_peoples_number = random.randint(1, max_add_peoples_mumber)
    add_people_index = 0
    imgHid1, imgCid1, imgIid1, jsondata1, canvas1 = readallinformation(rootpath, imagepath, fileindex1)
    newHid, newCid, newIid, newjsondata, newimg = imgHid1, imgCid1, imgIid1, jsondata1, canvas1
    information = []
    information.append([fileindex1, 0, 0, 0])
    while add_people_index < add_peoples_number:

        success, newHid, newCid, newIid, newjsondata, newimg, fileindex2, person2, deltax, deltay = people_enhance_once(
            newHid, newCid, newIid, newjsondata, newimg)
        if success:
            add_people_index = add_people_index + 1
            information.append([fileindex2, person2, deltax, deltay])
            # print('add_people_index,add_peoples_number:',add_people_index,add_peoples_number)
    return success, newHid, newCid, newIid, newjsondata, newimg, information


if __name__ == '__main__':
    # root = 'C:/Users/LI/Desktop/learning/dataset/CIHP/'
    root = 'C:/Users/Yhc/Desktop/'
    rootpath_all = root + 'slice/'
    # writepath='C:/Users/LI/Desktop/learning/dataset/CIHP/instance-level_human_parsing/tandv/augmentation/people/'
    writepath = 'C:/Users/Yhc/Desktop/slice/augmentation/'

    rootpath = rootpath_all
    imagepath = rootpath + 'Images/'
    fileindexs, imagenames = readimagename(rootpath_all,"train_id")
    print("file: ", fileindexs)
    print("image: ", imagenames)
    imagenumber = len(fileindexs)
    people_enhance_index = 0

    while people_enhance_index < people_enhance_number:
        imgindex1=random.randint(0,imagenumber-1)
        fileindex1=fileindexs[imgindex1]
        # fileindex1 = 28517
        # fileindex1=random.randint(0,500)
        # print('fileindex1:',fileindex1)

        success, newHid, newCid, newIid, newjsondata, newimg, information = people_enhance(fileindex1, rootpath,
                                                                                           imagepath)

        if success:
            fileindexnew = 10001 + people_enhance_index
            # print('to write jason')
            writecrowd(newHid, newCid, newIid, newjsondata, newimg, writepath, fileindexnew, information)
            people_enhance_index = people_enhance_index + 1
            print(fileindex1, len(information) - 1, people_enhance_index, 'done')
