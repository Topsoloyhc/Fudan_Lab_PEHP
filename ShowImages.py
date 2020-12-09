import sys

import cv2
import numpy as np

# 定义函数，第一个参数是缩放比例，第二个参数是需要显示的图片组成的元组或者列表
def ManyImgs(scale, imgarray):
    rows = len(imgarray)         # 元组或者列表的长度
    cols = len(imgarray[0])      # 如果imgarray是列表，返回列表里第一幅图像的通道数，如果是元组，返回元组里包含的第一个列表的长度
    # print("rows=", rows, "cols=", cols)

    # 判断imgarray[0]的类型是否是list
    # 是list，表明imgarray是一个元组，需要垂直显示
    rowsAvailable = isinstance(imgarray[0], list)

    # 第一张图片的宽高
    width = imgarray[0][0].shape[1]
    height = imgarray[0][0].shape[0]
    # print("width=", width, "height=", height)

    # 如果传入的是一个元组
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # 遍历元组，如果是第一幅图像，不做变换
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                # 将其他矩阵变换为与第一幅图像相同大小，缩放比例为scale
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)
                # 如果图像是灰度图，将其转换成彩色显示
                if  len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        # 创建一个空白画布，与第一张图片大小相同
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows   # 与第一张图片大小相同，与元组包含列表数相同的水平空白图像
        for x in range(0, rows):
            # 将元组里第x个列表水平排列
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)   # 将不同列表垂直拼接
    # 如果传入的是一个列表
    else:
        # 变换操作，与前面相同
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # 将列表水平排列
        hor = np.hstack(imgarray)
        ver = hor
    return ver

if __name__ == '__main__':
    # 读取图片
    image = cv2.imread("slice/Images/0010001.jpg")
    human = cv2.imread("slice/Human_ids/0010001.png")
    instance= cv2.imread("slice/Instances/0010001.png")
    instance_id= cv2.imread("slice/Instance_ids/0010001.png")
    categorie = cv2.imread("slice/Categories/0010001.png")

    # 第二种方法，调用我们自己编写的一个函数实现任意大小、任意通道数的图像堆叠显示
    # 以原来的0.8倍并排显示两张图片
    # stackedimageh = ManyImgs(0.8, ([imgflower, imgwomen]))
    #
    # # 垂直显示
    # stackedimagev = ManyImgs(0.8, ([imgflower, imgwomen], [imgwomen, imgflower]))

    # 如果只有图片数量为奇数，并希望能够垂直显示，可以创建一个空白图像
    Blankimg = np.zeros((200, 200), np.uint8)  # 大小可以任意函数会将其强制转换
    stackedimageb = ManyImgs(0.8, ([image, human, instance]))

    # cv2.namedWindow("stackedimageh")
    # cv2.imshow("stackedimageh", stackedimageh)
    #
    # cv2.namedWindow("stackedimagev")
    # cv2.imshow("stackedimagev", stackedimagev)

    cv2.namedWindow("stackedimageb")
    cv2.imshow("stackedimageb", stackedimageb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


