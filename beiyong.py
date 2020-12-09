# coding=utf-8

import json
import pathlib

import testCode
import numpy as np
import cv2
from PIL import Image

testimage = cv2.imread('slice/Images/0000006.jpg')
testHid = np.array(Image.open(np.str('slice/Human_ids/0000006.png')))
testinstance = np.array(Image.open(np.str('slice/Instances/0000006.png')))
fileindexs = ['6', '7', '8']
newjsondata = {}
newjsondata['Test'] = {
    'test': True
}

def test_cropInstance_testcases():
    mask1 = (testHid == 1)
    mask2 = (testHid == 2)

    newBbox1, oldBbox1, canvas1, instance1, imgHid1 = testCode.Cropper.cropInstance(mask1, testimage, testinstance, testHid)

    newBbox2, oldBbox2, canvas2, instance2, imgHid2 = testCode.Cropper.cropInstance(mask2, testimage, testinstance, testHid)

    print("newBbox:", newBbox1, newBbox2)
    print("oldBbox:", oldBbox1, oldBbox2)
    print("img shap:", canvas1.shape, canvas2.shape)
    if newBbox1[2] == canvas1.shape[0] and newBbox1[3] == canvas1.shape[1]:
        print("img1 w and h True")
    else:
        print("img1 w or h Not True")

def test_calculateXY_testcases():
    xmin, ymin, xmax, ymax = 112, 195, 170, 458
    xmin, ymin, neww, newh, oldBbox = testCode.Cropper.calculateXY(testimage, xmin, ymin, xmax, ymax)

    print(xmin, ymin, xmax, ymax, oldBbox)

def test_saveImgs_testcases():
    testCode.Cropper.saveImgs(testHid, newjsondata,testimage, testinstance, '0000006-test')
    path = pathlib.Path('slice/crop/Images/0000006-test.jpg')
    if path.is_file():
        print('save succeed')
    else:
        print('save failed')


if __name__ == "__main__":
    test_calculateXY_testcases()