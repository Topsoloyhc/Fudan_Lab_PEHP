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


def writejson_first(fileindex, rootpath, outjsonfile, fileindexs, models):
    filename = "%07d" % fileindex
    imagepath = rootpath + 'Images/' + filename + '.jpg'
    output_image = rootpath + 'vis/' + filename + '.jpg'
    imageHid = rootpath + 'Human_ids/' + filename + '.png'
    imageCid = rootpath + 'Category_ids/' + filename + '.png'
    imageIid = rootpath + 'Instance_ids/' + filename + '.png'
    imgHid, imgCid, imgIid = read.readCIHP(fileindex, rootpath)
    instance_number, resolution, partlabels = myjson.writejson_2(imgHid, imgCid, imgIid)
    dictwrite = {'Filename': filename,
                 'Imagepath': imagepath,
                 'OutputImagepath': output_image,
                 'Human_ids_path': imageHid,
                 'Humansimg_path': rootpath + 'Humans/' + filename + '.png',
                 'Category_ids_path': imageCid,
                 'Categoriesimg_path': rootpath + 'Categories/' + filename + '.png',
                 'Instance_ids_path': imageIid,
                 'Instancesimg_path': rootpath + 'Instances/' + filename + '.png',

                 # TODO
                 'Is_train': True,
                 'Is_val': False,
                 'Is_test': False,

                 'Resolution': resolution,
                 'IoU': 0,
                 'Instance_number': int(instance_number),
                 'Visible_Parsing': int(instance_number),
                 'Parts_number': int(partlabels.size),
                 'Parts': partlabels.tolist(),
                 'Keypoints_number': 17,
                 'Visible_keypoints_number': 0,
                 'Visible_keypoints_type': [],
                 'Visible_keypoints_type_number': 0,

                 # TODO
                 'Visible_Pose': 0,
                 'Instances': dict(),
                 # TODO
                 'Delete': {
                     'Is_deleted': False,
                     'Reason': "",
                 },
                 # TODO
                 'DataAugmentation': {
                     'Has_Augmentation': {
                         'Has_Augmentation': False,
                         'Background_Augmentation': {
                             'Has': False,
                             'New_Imagepath': []
                         },
                         'Occlusion_Augmentation': {
                             'Has': False,
                             'New_Imagepath': []
                         },
                         'Part_Augmentation': {
                             'Has': False,
                             'New_Imagepath': []
                         },
                     },
                     'Is_Augmentation': {
                         'Is_Augmentation': False,
                         'Background_Augmentation': False,
                         'Occlusion_Augmentation': False,
                         'Part_Augmentation': False,
                         'Original_Imagepath': None,
                     },

                 },
                 # TODO
                 'TestSet': {
                     'In_Simple_Parts_Testset': False,
                     'In_Normal_Parts_Testset': False,
                     'In_Hard_Parts_Testset': False,
                     'In_Not_Crowded_Testset': False,
                     'In_Crowded_Testset': False,
                     'In_Scatter_Testset': False,
                     'In_Occlude_Testset': False,
                 },

                 }
    return dictwrite


def writejson_2(imgHid, imgCid, imgIid):
    instance_number = imgHid.max()
    resolution = (imgHid.shape[0], imgHid.shape[1])
    partlabels = np.unique(imgCid)
    return instance_number, resolution, partlabels


def writejson_instance(human_index, imgHid, imgCid, resolution):
    instance_name = 'Instance' + str(human_index)
    bbox, bboxResolution = cal.cal_bbox(imgHid, human_index)
    bboxScale = float(bboxResolution[0] * bboxResolution[1]) / float(resolution[0] * resolution[1])
    segmentationResolution = int(np.sum(imgHid == human_index))
    segmentationScale = float(segmentationResolution) / float(resolution[0] * resolution[1])
    partlabels_instance = np.unique(imgCid[imgHid == human_index])
    instance = {'Instance_name': instance_name,
                'Has_parsing': True,
                'Bbox': bbox,
                'BboxResolution': bboxResolution,
                'SegmentationResolution': segmentationResolution,
                'BboxScale': bboxScale,
                'IoU': 0,
                'SegmentationScale': segmentationScale,
                'Parts_number': int(partlabels_instance.size),
                'Parts': partlabels_instance.tolist(),
                'Poses_of_predicted_number': int(0),
                'Poses_of_predicted': dict(),
                'Has_gtpose': False,
                'Gtpose': dict()
                }
    # print(instance)
    return instance


def writejson_instances(dictwrite, imgHid, imgCid, resolution):
    instance_number = dictwrite['Instance_number']
    # print(instance_number)

    for human_index in range(1, instance_number + 1):
        instance_name = 'Instance' + str(human_index)
        instance = myjson.writejson_instance(human_index, imgHid, imgCid, resolution)
        dictwrite['Instances'][instance_name] = instance
    return dictwrite


def writejson_segiou(dictwrite, imgHid):
    instance_number = dictwrite['Instance_number']
    # print(instance_number)
    meanmeaniou = 0
    meanmeansegmentiou = 0
    meanmeanalliou = 0
    meanmeanallsegiou = 0
    for human_index1 in range(1, instance_number + 1):
        meaniou = 0
        meansegmentiou = 0
        meanalliou = 0
        meanallsegiou = 0
        exist_occlusion = 0
        existsegment_occlusion = 0
        instance_name1 = 'Instance' + str(human_index1)
        bbox1 = dictwrite['Instances'][instance_name1]['Bbox']
        mask1 = (imgHid == human_index1)
        for human_index2 in range(1, instance_number + 1):
            if human_index2 == human_index1:
                continue

            instance_name2 = 'Instance' + str(human_index2)

            bbox2 = dictwrite['Instances'][instance_name2]['Bbox']
            mask2 = (imgHid == human_index2)
            iou = cal.compute_iou(bbox1, bbox2)
            segmentiou = cal.compute_segiou(bbox1, bbox2, mask1, mask2)
            if iou > 0:
                meaniou = meaniou + iou
                exist_occlusion = exist_occlusion + 1
            if segmentiou > 0:
                meansegmentiou = meansegmentiou + segmentiou
                existsegment_occlusion = existsegment_occlusion + 1

            meanalliou = meanalliou + iou
            meanallsegiou = meanallsegiou + segmentiou
            # meaniou=meaniou+iou
        # meaniou=meaniou/float(instance_number-1)
        if instance_number - 1:
            meanalliou = meanalliou / float(instance_number - 1)
            meanallsegiou = meanallsegiou / float(instance_number - 1)

        if exist_occlusion:
            meaniou = meaniou / float(exist_occlusion)
        if existsegment_occlusion:
            meansegmentiou = meansegmentiou / float(existsegment_occlusion)
        dictwrite['Instances'][instance_name1]['IoU'] = meaniou
        dictwrite['Instances'][instance_name1]['SegIoU'] = meansegmentiou
        dictwrite['Instances'][instance_name1]['allIoU'] = meanalliou
        dictwrite['Instances'][instance_name1]['allSegIoU'] = meanallsegiou
        # print(meaniou)
        meanmeaniou = meanmeaniou + meaniou
        meanmeansegmentiou = meanmeansegmentiou + meansegmentiou
        meanmeanalliou = meanmeanalliou + meanalliou
        meanmeanallsegiou = meanmeanallsegiou + meanallsegiou

    meanmeaniou = meanmeaniou / float(instance_number)
    meanmeansegmentiou = meanmeansegmentiou / float(instance_number)
    meanmeanalliou = meanmeanalliou / float(instance_number)
    meanmeanallsegiou = meanmeanallsegiou / float(instance_number)

    dictwrite['IoU'] = meanmeaniou
    dictwrite['SegIoU'] = meanmeansegmentiou
    dictwrite['allIoU'] = meanmeanalliou
    dictwrite['allSegIoU'] = meanmeanallsegiou
    return dictwrite
