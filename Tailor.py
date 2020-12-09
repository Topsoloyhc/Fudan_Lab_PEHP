import json
import numpy as np
import cv2
from PIL import Image
import myjson

#按照每个实例的bbox（长宽增加30%）裁剪出来，保存：1、裁剪后的图；2、Json文件（原图、相对于原图的坐标位置、裁剪后的图片大小）
#暂定为显示原图（image、Categories、Human、Instances）、裁剪后的图
#shap原图尺寸、canvas图片、shap原图尺寸、mask 0，1掩码、imgCid类别掩码
#human：每个人的像素不一样；instance：每个实例的每个部位像素不一样；category：同一类别（19个）的像素是一样的，不同类别像素不同
#按照human的mask来把人的那部分抠出来，背景变成空白


# 按照bbox长宽增大30%做一个裁剪,需要经过裁剪的实例，修改后保存下的内容（shap、mask、imgCid、imgHid）
def tailorPeople(canvas, instance, mask, imgCid, imgHid, imgIid, Bbox):
    #imgHid.shape[0], imgHid.shape[1]
    findresult = np.where(mask)
    xmin, ymin, xmax, ymax, w, h = 0, 0, 0, 0, 0, 0
    if findresult[0].size > 0:
        ymin = int(findresult[0].min())
        ymax = int(findresult[0].max())
        xmin = int(findresult[1].min())
        xmax = int(findresult[1].max())
    w = xmax - xmin + 1
    h = ymax - ymin + 1

    percentage = 0.15
    ymin = max(0, int(ymin - h * percentage))
    ymax = min(canvas.shape[0], int(ymax + h * percentage))
    xmin = max(0, int(xmin - w * percentage))
    xmax = min(canvas.shape[1], int(xmax + w * percentage))
    neww = xmax - xmin
    newh = ymax - ymin

    Bbox = [xmin, ymin, neww, newh]
    # print("imgCid", imgCid.shape)
    # print("canvas", canvas.shape)
    canvas = canvas[ymin:ymax, xmin:xmax] #ymin->ymax, xmin->xmax（范围）
    instance = instance[ymin:ymax, xmin:xmax]
    mask = mask[ymin:ymax, xmin:xmax]
    imgCid = imgCid[ymin:ymax, xmin:xmax]
    imgHid = imgHid[ymin:ymax, xmin:xmax]
    imgIid = imgIid[ymin:ymax, xmin:xmax]
    # print("change imgCid", imgCid.shape)
    # print("change canvas", canvas.shape)
    # for i in range(0, canvas.shape[0]):
    #     for j in range(0, canvas.shape[1]):
    #         if mask[i][j] != True:
    #             canvas[i][j] = [255, 255, 255]
    canvas[mask != True] = [255, 255, 255]
    instance[mask != True] = [255, 255, 255]
    return canvas, instance, mask, imgCid, imgHid, imgIid, Bbox


#生成新的图片 + 保存相关的Json数据
def writeCrowd(newHid, newCid, newIid, newjsondata, newimg, newinstance, writepath, fileindexnew):
    cv2.imwrite(writepath + 'Images/' + fileindexnew + '.jpg', newimg)
    # cv2.imwrite(writepath + 'Instances/' + fileindexnew + '.png', newinstance)
    image = Image.fromarray(newinstance)
    image.save(newjsondata['Instance_path'])
    image = Image.fromarray(newHid)
    image.save(newjsondata['Human_ids_path'])
    image = Image.fromarray(newCid)
    image.save(newjsondata['Category_ids_path'])
    image = Image.fromarray(newIid)
    image.save(newjsondata['Instance_ids_path'])
    writeJsonFile(newjsondata, writepath + 'jsonfile/json' + fileindexnew + '.json')


 #以dict的形式保存json数据，然后写入到outjsonfile处
def writeJsonFile(dictwrite, outjsonfile):
    newjsondata = json.dumps(dictwrite, indent=4) #用参数indent=4来对json进行数据格式化输出，美化输出
    with open(outjsonfile, 'w') as fw:
        fw.write(newjsondata)

#获取所有的图片的名称
def readImgName(rootpath, TYPE):
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

#获取某一图片的相关信息
def readOneImg(rootpath, imagepath, instancepath, fileindex):
    image = imagepath + "%07d" % fileindex + '.jpg'
    instance = instancepath + "%07d" % fileindex + '.png'
    imgHid, imgCid, imgIid, instance = readCIHP(fileindex, rootpath)
    canvas = cv2.imread(image)
    # instance = cv2.imread(instance)
    return imgHid, imgCid, imgIid, canvas, instance

#获取图片相关的json数据
def readImgJson(rootpath, fileindex):
    jsondata = readJson(rootpath + 'jsonfile/json' + "%07d" % fileindex + '.json')
    return jsondata

#获取json数据
def readJson(injsonfile):
    with open(injsonfile, 'rb') as f:
        jsondata = json.load(f)
    return jsondata

#从数据集中获取数据，包括Hid，Cid，Iid
def readCIHP(fileindex, rootpath):
    name = "%07d" % fileindex
    imageHid = rootpath + 'Human_ids/' + name + '.png'
    imageCid = rootpath + 'Category_ids/' + name + '.png'
    imageIid = rootpath + 'Instance_ids/' + name + '.png'
    instance = rootpath + 'Instances/' + name + '.png'
    instance = np.array(Image.open(np.str(instance)))
    imgHid = np.array(Image.open(np.str(imageHid)))
    imgCid = np.array(Image.open(np.str(imageCid)))
    imgIid = np.array(Image.open(np.str(imageIid)))
    return imgHid, imgCid, imgIid, instance

#保存裁剪后的图像的相关json数据
def saveJson(person, rootpath, imagepath, fileindex, fileindexnew, oldBbox, newBbox):
    newjsondata = {}
    newjsondata['Filename'] = fileindexnew
    newjsondata['Imagepath'] = writepath + 'Images/' + fileindexnew + '.jpg'
    newjsondata['Instance_path'] = writepath + 'Instances/' + fileindexnew + '.png'
    newjsondata['Human_ids_path'] = writepath + 'Human_ids/' + fileindexnew + '.png'
    newjsondata['Category_ids_path'] = writepath + 'Category_ids/' + fileindexnew + '.png'
    newjsondata['Instance_ids_path'] = writepath + 'Instance_ids/' + fileindexnew + '.png'

    newjsondata['OrignalImagepath'] = imagepath + "%07d" % fileindex + '.jpg'
    newjsondata['OrignalHuman_ids_path'] = rootpath + 'Human_ids/' + "%07d" % fileindex + '.png'
    newjsondata['OrignalCategory_ids_path'] = rootpath + 'Category_ids/' + "%07d" % fileindex + '.png'
    newjsondata['OrignalInstance_ids_path'] = rootpath + 'Instance_ids/' + "%07d" % fileindex + '.png'

    newjsondata['Bbox'] = {
        'Orginal' : oldBbox,
        'New' : newBbox
    }
    newjsondata['OrginalCoordinate'] = { #在原图中的坐标位置
        'Xmin' : newBbox[0],
        'Ymin' : newBbox[1]
    }
    newjsondata['Resolution'] = [newBbox[2], newBbox[3]]

    newjsondata['DataTailor'] = {
        'Type': "Tailor_instance",
        'Original_Image': imagepath + "%07d" % fileindex + '.jpg',
        'Instance_Index': person
    }

    return  newjsondata

def writeTxt(writepath, fileindexnew):
    f = open(writepath + 'train_id.txt', mode='a')  # 打开文件，若文件不存在系统自动创建。
    # 参数name 文件名，mode 模式    # w 只能操作写入  r 只能读取   a 向文件追加
    # w+ 可读可写   r+可读可写    a+可读可追加   # wb+写入进制数据 # w模式打开文件，如果文件中有数据，再次写入内容，会把原来的覆盖掉

    f.write(fileindexnew + '\n')  # write 写入

    # f.writelines(['hello\n', 'world\n', '你好\n', '世界\n'])  # writelines()函数 会将列表中的字符串写入文件中，但不会自动换行，如果需要换行，手动添加换行符
    # # 参数 必须是一个只存放字符串的列表
    f.close()  # 关闭文件

if __name__ == '__main__':
    # root = 'C:/Users/LI/Desktop/learning/dataset/CIHP/'
    rootpath = 'slice/'
    # writepath='C:/Users/LI/Desktop/learning/dataset/CIHP/instance-level_human_parsing/tandv/augmentation/people/'
    writepath = 'slice/tailor/'

    imagepath = rootpath + 'Images/'
    instancepath = rootpath + 'Instances/'
    fileindexs, imagenames = readImgName(rootpath,"train_id")
    imagenumber = len(fileindexs)
    print("file: ", fileindexs)
    print("image: ", imagenames)
    # print("imagenumber: ", imagenumber)
    index = 1

    f = open(writepath + 'train_id.txt', mode='a')
    f.seek(0)
    f.truncate()  # 清空文件

    for img in range(0, imagenumber):#对每一张图片进行裁剪
        print(fileindexs[img])
        #获取要裁剪的图片的信息
        fileindex = fileindexs[img]
        jsondata = readImgJson(rootpath, fileindex)
        personNum = jsondata['Instance_number']  # 一个图像中有多少个实例（人）
        for person in range(1, personNum + 1):
            #要在这里面获取图片相关的数据，for循环里
            imgHid, imgCid, imgIid, canvas, instance = readOneImg(rootpath, imagepath, instancepath, fileindex)
            originalcanvas, originalimgCid, orginalimgHid, orginalimgIid, orginalinstance = canvas, imgCid, imgHid, imgIid, instance
            oldBbox = jsondata['Instances']['Instance' + str(person)]['Bbox'] #获取每个实例的Bbox
            print(oldBbox)

            mask = (imgHid == person)

            print(imgHid.shape[0], imgHid.shape[1])
            canvas_t, instance_t, mask_t, imgCid_t, imgHid_t, imgIid_t, newBbox_t = tailorPeople(originalcanvas, orginalinstance, mask,
                                                                           originalimgCid, orginalimgHid, orginalimgIid, oldBbox)
            print("changeBbox：", newBbox_t)
            fileindexnew = str(fileindexs[img]) + '_' + str(person).rjust(2, '0')
            #原字符串右侧对齐， 左侧补零: str.rjust(width, '0')
            fileindexnew = fileindexnew.rjust(10, '0')
            newjsondata = saveJson(person, rootpath, imagepath, fileindex, fileindexnew, oldBbox, newBbox_t)
            writeCrowd(imgHid_t, imgCid_t, imgIid_t, newjsondata, canvas_t, instance_t, writepath, fileindexnew)
            writeTxt(writepath, fileindexnew)
            print("原图: ", fileindexnew, "第", index, "个实例裁剪", 'done')
            index += 1



