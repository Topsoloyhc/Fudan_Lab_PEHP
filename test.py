import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def __writeTxt(fileindexnew):
    writepath = 'crop/'
    f = open(writepath + 'train_id.txt', mode='a')
    f.write(fileindexnew + '\n')  # write 写入

def __readImgName():
    path = 'slice/crop/train_id.txt'
    allfile = np.loadtxt(path, delimiter=',')
    # fileindexs = []
    # for i in allfile:
    #     inputid = int(i)
    #     fileindexs.append(inputid)
    # return fileindexs
    return  allfile


if __name__ == "__main__":
    # files = __readImgName()
    # # files = files[21457:]
    # print(files)
    # for i in files:
    #     inputid = str("%09d" % i)
    #     print(inputid)
    #     name = inputid[:7] + '_' + inputid[7:]
    #     print(name)
    imgid = 114801
    imgname = "%07d" % (imgid // 100) + '_' + "%02d" % (imgid % 100)
    print(imgname)

    imgnames = []
    with open('slice/crop/train_id.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            imgnames.append(line)
    print(len(imgnames))

    # f = open('train_id.txt', mode='a')
    # f.seek(0)
    # f.truncate()  # 清空文件
    # for fi in files:
    #     print(fi)
    #     __writeTxt(str(fi))
    # s = 10
    # print(str(s).rjust(2, '0').rjust(10, '0'))
