# coding=utf-8

import json
import pathlib

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import CropJson


class Cropper(object):

    def __init__(self, config):
        self.__config = config
        # print(config['rootpath'])

    def crop(self):
        fileindexs = self.__readImgName("train_id")
        imgnum = len(fileindexs)
        if self.__config['is_reCrop']:
            f = open(self.__config['writepath'] + 'train_id.txt', mode='a')
            f.seek(0)
            f.truncate()  # 清空txt文件
            for img in tqdm(range(imgnum)):
                fileindex = fileindexs[img]
                self.__cropOneImg(fileindex)
        else:
            for img in tqdm(range(imgnum)):
                fileindex = fileindexs[img]
                if self.__isCrop(fileindex):
                    pass
                else:
                    # canvas, instance, imgHid = self.__getInfo(fileindex)
                    self.__cropOneImg(fileindex)

            # print("完成对" + "%07d" % fileindex + ".jpg的实例分割")

    def __getInfo(self, fileindex):
        name = "%07d" % fileindex
        image = self.__config['imagepath'] + name + '.jpg'
        imageHid = self.__config['rootpath'] + 'Human_ids/' + name + '.png'
        instance = self.__config['rootpath'] + 'Instances/' + name + '.png'
        canvas = cv2.imread(image)
        # instance = cv2.imread(instance)
        # imgHid = cv2.imread(imageHid)
        instance = np.array(Image.open(np.str(instance)))
        imgHid = np.array(Image.open(np.str(imageHid)))
        return canvas, instance, imgHid

    def __readImgName(self, txtName):
        path = self.__config['rootpath'] + txtName + '.txt'
        allfile = np.loadtxt(path, delimiter=',')
        fileindexs = []
        for i in allfile:
            inputid = int(i)
            fileindexs.append(inputid)
        return fileindexs

    def __cropOneImg(self, fileindex):
        canvas, instance, imgHid = self.__getInfo(fileindex)
        instanceNum = len(np.unique(imgHid))
        for instance_index in range(1, instanceNum):
            oricanvas, oriinstance, oriimgHid = self.__getInfo(fileindex)
            fileindexnew = str(fileindex) + '_' + str(instance_index).rjust(2, '0')
            # 原字符串右侧对齐， 左侧补零: str.rjust(width, '0')
            fileindexnew = fileindexnew.rjust(10, '0')
            mask = (oriimgHid == instance_index)
            newBbox, oldBbox, canvas, instance, imgHid = self.__cropInstance(mask, oricanvas, oriinstance, oriimgHid)
            jasondata = CropJson.saveJson(instance_index, fileindex, fileindexnew, oldBbox, newBbox)
            self.__saveImgs(imgHid, jasondata, canvas, instance, fileindexnew)
            self.__writeTxt(fileindexnew)

    def __cropInstance(self, mask, canvas, instance, imgHid):
        findresult = np.where(mask)
        xmin, ymin, xmax, ymax, w, h = 0, 0, 0, 0, 0, 0
        newBbox, oldBbox = [0, 0, 0, 0], [0, 0, 0, 0]
        if findresult[0].size > 0:
            ymin = int(findresult[0].min())
            ymax = int(findresult[0].max())
            xmin = int(findresult[1].min())
            xmax = int(findresult[1].max())
            xmin, ymin, neww, newh, oldBbox = self.__calculateXY(canvas, xmin, ymin, xmax, ymax)
            newBbox = [xmin, ymin, neww, newh]
        canvas = canvas[ymin:ymin + newh, xmin:xmin + neww]  # ymin->ymax, xmin->xmax（范围）
        instance = instance[ymin:ymin + newh, xmin:xmin + neww]
        mask = mask[ymin:ymin + newh, xmin:xmin + neww]
        imgHid = imgHid[ymin:ymin + newh, xmin:xmin + neww]
        canvas[~mask] = [0, 255, 0]
        instance[~mask] = [255, 255, 255]
        return newBbox, oldBbox, canvas, instance, imgHid

    def __calculateXY(self, canvas, xmin, ymin, xmax, ymax):
        percentage = self.__config['ratio']
        w, h = xmax - xmin + 1, ymax - ymin + 1
        oldBbox = [xmin, ymin, w, h]
        ymin = max(0, int(ymin - h * percentage))
        ymax = min(canvas.shape[0], int(ymax + h * percentage))
        xmin = max(0, int(xmin - w * percentage))
        xmax = min(canvas.shape[1], int(xmax + w * percentage))
        neww, newh = xmax - xmin, ymax - ymin
        return xmin, ymin, neww, newh, oldBbox

    def __saveImgs(self, newHid, newjsondata, newimg, newinstance, fileindexnew):
        writepath = self.__config['writepath']
        cv2.imwrite(writepath + 'Images/' + fileindexnew + '.jpg', newimg)
        image = Image.fromarray(newinstance)
        image.save(newjsondata['Output']['Instance_path'])
        image = Image.fromarray(newHid)
        image.save(newjsondata['Output']['Human_ids_path'])
        self.__writeJsonFile(newjsondata, writepath + 'jsonfile/json' + fileindexnew + '.json')

    def __writeJsonFile(self, dictwrite, outjsonfile):
        newjsondata = json.dumps(dictwrite, indent=4)  # 用参数indent=4来对json进行数据格式化输出，美化输出
        with open(outjsonfile, 'w') as fw:
            fw.write(newjsondata)

    def __writeTxt(self, fileindexnew):
        writepath = self.__config['writepath']
        f = open(writepath + 'train_id.txt', mode='a')
        f.write(fileindexnew + '\n')  # write 写入

    def __isCrop(self, fileindex):
        path = pathlib.Path(self.__config['writepath'] + '/Images/' + "%07d" % fileindex + "_01.jpg")
        return path.is_file()


if __name__ == "__main__":
    config = {
        'rootpath': 'slice/',
        'writepath': 'slice/crop/',
        'imagepath': 'slice/Images/',
        'instancepath': 'slice/Instances/',
        'ratio': 0.15,
        'is_reCrop':True,
    }
    Cropper(config).crop()
