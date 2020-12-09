# coding=utf-8


import pathlib
import numpy as np

#simplepose（17*2 + 1）中检测到17个关节点，共有35个值，最后一个表示姿态估计结果得到的分数
#keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[-2]))  # s[19] is the score
#有的图像过大，没有办法处理，没有得到姿态估计结果
# self.parts = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
#               'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank'] 共17个节点+ neck? navel?

#personlab（17*3）中检测到17个关节点，前两个为坐标x，y
#hash_pl = [0, 14, 13, 16, 15, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]  # PL->SP对应的顺序修改
#存在有些图像下检测到多个骨架，有些未检测到骨架
#KEYPOINTS = ["nose", "Rshoulder", "Relbow", "Rwrist",  "Lshoulder", "Lelbow", "Lwrist", "Rhip",
#             "Rknee", "Rankle", "Lhip", "Lknee", "Lankle", "Reye", "Leye", "Rear", "Lear"] 共17个节点 + "neck"

#Darkpose
#"keypoints": ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow",
# "right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
from tqdm import tqdm


class Voting(object):

    def __init__(self, config):
        self.__config = config

    # 主投票函数，循环执行n张图的结果
    def vote(self):
        # sppose, plpose, dppose =self.__getAllInfo( '0000006_01')
        # print(sppose,'\n',plpose,'\n',dppose)
        # pose, flag = self.__voteOnce(sppose, plpose, dppose)
        # print("pose:",pose)
        if(self.__config['is_revote']):
            f = open(self.__config['posepath']  + 'save_id.txt', mode='a')
            f.seek(0)
            f.truncate()
            f = open(self.__config['posepath'] + 'manual_id.txt', mode='a')
            f.seek(0)
            f.truncate()
        fileindexs = self.__readImgName()
        for imgpose in tqdm(fileindexs):
            sppose, plpose, dppose = self.__getAllInfo(imgpose)
            pose, flag = self.__voteOnce(sppose, plpose, dppose)
            self.__savePose(imgpose, pose, flag)

    # 针对某一张图的姿态估计结果进行一次投票选择结果 (存在某一种方法没有姿态估计结果的情况，所得到的 pose=[] 要进行判断)
    def __voteOnce(self, sppose, plpose, dppose):
        pose = []
        flag = True
        if sppose and plpose and dppose: #各个算法都存在姿态估计结果才进入投票
            for i in range(17):  # 对17个节点遍历
                pose.append(self.__jointCaculate(sppose[i], plpose[i], dppose[i]))
        else:
            flag = False
        return pose, flag

    # 计算两个坐标之间的距离
    def __distance(self, m, n):
        return np.sqrt(np.sum((np.array(m) -  np.array(n)) ** 2))

    # 每个关节点之间的计算，保留最好的估计结果的关键点
    def __jointCaculate(self, j1, j2, j3):
        xy_ave = [(j1[0] + j2[0] + j3[0]) / 3, (j1[1] + j2[1] + j3[1]) / 3]
        l1, l2, l3 = self.__distance(j1, xy_ave), self.__distance(j2, xy_ave), self.__distance(j3, xy_ave)
        #如果各个方法与平均点之间的距离小于10，则采用平均点位置作为关节点坐标，否则选取距离最短的那个关节点作为关节点坐标
        if (l1 + l2 + l3) / 3 < 10:
            return xy_ave
        else:
            lmin = min(l1, l2, l3)
            if lmin == l1:
                return j1
            if lmin == l2:
                return j2
            if lmin == l3:
                return j3

    # 将17*3的形式得到17*2的结果 适合PL、DP
    def __threeTotwo(self, pose):
        resultpose = []
        x = 0
        for i in range(17):
            resultpose.append([pose[x], pose[x + 1]])
            x += 3
        return resultpose

    # 关节点位置统一化，将姿态关节点顺序转换成SP对应的顺序
    def __poseTosp(self, pose, hash_index):
        tosppose = []
        for index in hash_index:
            tosppose.append(pose[index])
        return tosppose

    # 将从txt中得到的关节点信息变成[x,y]一对一对共17对的形式(17*2) --前提就是17*2的形式，适合SP
    def __tojoints(self, pose):
        resultpose = []
        for i in range(17):
            resultpose.append([pose[i], pose[i+1]])
        return resultpose

    # 获取train_id中各图像的名称，返回为0000006_01的形式
    def __readImgName(self):
        allfile = np.loadtxt(self.__config['rootpath'] + 'train_id.txt', delimiter=',')
        fileindexs = []
        for i in allfile:
            inputid = str("%09d" % i)
            name = inputid[:7] + '_' + inputid[7:]
            fileindexs.append(name)
        return fileindexs

    def __getAllInfo(self, imgpose):#要在这里判断是否有得到相应的姿态估计结果，且personlab中得到的有多个骨架的结果（如何处理，暂时只返回第一个姿态）
        hash_pl = [0, 14, 13, 16, 15, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]  # PL->SP对应的顺序修改
        #hash_dp = []  # DP->SP对应的顺序修改,无需修改，一样的顺序
        SPfilepath = self.__config['SPposepath'] + imgpose + '.txt'
        PLfilepath = self.__config['PLposepath'] + imgpose + '.txt'
        # HRfilepath = self.__config['HRposepath'] + imgpose + '.txt'
        DPfilepath = self.__config['DPposepath'] + imgpose + '.txt'
        sppose, plpose, dppose = [], [], []
        if(self.__isExit(SPfilepath)):
            print(SPfilepath + " SPpose exit")
            sppose = self.__tojoints(self.__getPose(self.__getInfo(SPfilepath)))
        if (self.__isExit(PLfilepath)):
            print(PLfilepath + " PLpose exit")
            plpose = self.__poseTosp(self.__threeTotwo(self.__getPose(self.__getInfo(PLfilepath))),hash_pl)
        #hrpose = self.__getInfo(HRfilepath)
        if (self.__isExit(DPfilepath)):
            print(DPfilepath + " DPpose exit")
            dppose = self.__threeTotwo(self.__getPose(self.__getInfo(DPfilepath)))

        return  sppose, plpose, dppose

    #SP：35个值，17*2+1；PL：51个值，17*3；#获取不同姿态估计结果得到的txt文件中的信息
    def __getInfo(self, filepath):
        allfile = np.loadtxt(filepath, delimiter=',') #存在一个问题，由于有多个骨架，所以读出来的会是多个filindexs []、[]形式
        fileindexs = []
        for i in allfile:
            # inputid = i
            fileindexs.append(i)
        return fileindexs

    # 获取姿态估计结果，判断是否具有多个姿态估计结果,对多个姿态进行处理保留下一个（或者直接判定为检测效果不佳？或者进行怎么样的比对）
    def __getPose(self, pose):
        if len(pose) == 35 or len(pose) == 51:
            return pose
        else:
            return pose[0]

    # 判断是否存在该pose结果,传入类似././0000006_01.txt路径
    def __isExit(self, filepath):
        return pathlib.Path(filepath).is_file()

    # 保存vote之后的姿态估计结果，好的姿态保留(vote文件夹保存，save_id)，不好的记录下来（manual_id）留做人工标记
    def __savePose(self, imgpose, pose, flag): #写入文件id时info后面要加换行符
        #flag用来判断这个图像是否得到vote之后的结果，即记录是否需要进行人工标注
        if flag: #true,得到vote后的姿态
            if self.__isExit(self.__config['voteposepath'] + imgpose + '.txt'): #若已经存在这个图像的姿态估计结果，清空
                f = open(self.__config['voteposepath'] + imgpose + '.txt', mode='a')
                f.seek(0)
                f.truncate()
            self.__writeTxt(self.__config['posepath'], 'save_id.txt', imgpose + '\n') #写入得到vote姿态的文件名称
            np.savetxt(self.__config['voteposepath'] + imgpose + '.txt', pose, fmt='%f', delimiter=',')
            # self.__writeTxt(self.__config['voteposepath'], imgpose + '.txt', pose) #写入vote之后的姿态估计结果
        else:
            self.__writeTxt(self.__config['posepath'], 'manual_id.txt', imgpose + '\n')  # 写入需要人工标注的文件名称

    def __writeTxt(self, writepath, filename, info): #将数据写入txt文件中
        f = open(writepath + filename, mode='a')
        # f.write(info + '\n')  # write 写入
        f.write(info)  # write 写入

if __name__ == "__main__":
    config = {
        'SPposepath': 'slice/crop/poseresult/SPjoints/',
        'PLposepath': 'slice/crop/poseresult/PLjoints/',
        'DPposepath': 'slice/crop/poseresult/DPjoints/',
        'voteposepath': 'slice/crop/poseresult/vote/',
        'posepath': 'slice/crop/poseresult/',
        'rootpath': 'slice/crop/',
        'is_revote': True,
    }
    Voting(config).vote()