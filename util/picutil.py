import numpy as np
import random
import cv2
import copy
import os

def get_block_pixdiff(all_imgs,states,feature_number,edge = 30):
    '''
    通过点位置划分block，在每个block取pixdiff特征
    :param all_imgs:
    :param states:
    :param feature_number:
    :return:
    '''
    points =[]
    for state in states:
        points.append(label_to_landmark(state))

    features = pick_pixdiff_Feature(len(points[0]),feature_number,edge)
    values = extract_all_pixdiff(all_imgs,points,features,edge)
    return features,values




def pick_pixdiff_Feature(block_number,feature_number,edge):
    '''
    对于不同的block  ，随机生成点对
    :param block_number:
    :param feature_number:
    :param edge:
    :return:
    '''
    features = []
    for block_id in range(block_number):
        for i in range(feature_number):
            deta_x1 =random.randint(-edge,edge)
            deta_y1 = random.randint(-edge,edge)
            deta_x2 = random.randint(-edge,edge)
            deta_y2 = random.randint(-edge,edge)
            features.append([block_id,deta_x1,deta_y1,deta_x2,deta_y2])
    return features



def extract_all_pixdiff(all_imgs,landmark,features,edge):
    '''
    在图片给定的区域随机取点对相减作为特征值，
    :param all_imgs: 图片序列
    :param edgs: 左 右 下 上 四个边界的像素值[161, 825, 87, 744]
    :param feature_number: 要选几个点对
    :return: 返回点对的位置  和点对相减的值
    '''
    values = []
    for idx in range(len(all_imgs)):
        value = []

        blocklist = cut_img_by_landmark(all_imgs[idx], landmark[idx],edge)
        for feature in features:
            value.append(extract_single_pixdiff(blocklist,feature))
            #showpic(block,points)
        values.append(value)
    return values

def extract_single_pixdiff(blocklist,feature):
    '''
    提取一个block中的feature
    :param block:
    :param feature:
    :return:
    '''
    block = blocklist[feature[0]]
    points = []
    center = (int(block.shape[0] / 2), int(block.shape[1] / 2))
    x1 = int(feature[1])
    y1 = int(feature[2])
    x2 = int(feature[3])
    y2 = int(feature[4])
    single_value=int(block[y1 + center[0]][x1 + center[1]]) - int(block[y2 + center[0]][x2 + center[1]])
    points.append((x1 + center[1], y1 + center[0]))
    points.append((x2 + center[1], y2 + center[0]))
    return single_value


def cut_img_by_landmark(img, landmark, edge=30,blockid = None):
    '''
    输入一张图片，返回按照lardmark点切割的小图片块,edge x是小方块边长的一半，也就是距中心的距离
    :param img:
    :param landmark:
    :return:
    '''
    blocklist = []
    # showpic(img)
    shape = img.shape
    for idx, point in enumerate(landmark):
        block = img[point[1] - edge:point[1] + edge+1, point[0] - edge:point[0] + edge+1]

        # if block.shape[0]!=32 or block.shape[1]!=32:
        #     print("图片关键点溢出了边界")
        #     print(idx)
        #     showpic(img,landmark)
        #     print(block)
        #     showpic(block)
        if point[1] - edge < 0:
            block = img[0:point[1] + edge + 1, point[0] - edge:point[0] + edge + 1]
            print(np.zeros((0 - (point[1] - edge), 2 * edge+1)).shape)
            print(block.shape)
            print(point)
            print(edge)
            print(img.shape)

            block = np.vstack((np.zeros(((0 - (point[1] - edge)), 2 * edge+1)),block)).astype(np.uint8)
        if point[1] + edge + 1 > shape[0]:
            # print(block.shape)
            # print(np.zeros((point[1]+16-shape[0], 32)).shape)
            block = img[point[1] - edge:shape[0], point[0] - edge:point[0] + edge + 1]
            block = np.vstack((block, np.zeros((point[1] + edge+1 - shape[0], 2 * edge+1)))).astype(np.uint8)
            # print(block)
        if point[0] - edge < 0:
            block = img[point[1] - edge:point[1] + edge + 1, 0:point[0] + edge + 1]
            block = np.hstack((np.zeros((2 * edge+1, 0 - (point[0] - edge))), block)).astype(np.uint8)
        if point[0] + edge +1> shape[1]:
            block = img[point[1] - edge:point[1] + edge + 1, point[0] - edge:shape[1]]
            block = np.hstack((block, np.zeros((2 * edge+1, point[0] + edge+1 - shape[1])))).astype(np.uint8)

        blocklist.append(block)
        # showpic(img[point[1]-16:point[1]+16,point[0]-16:point[0]+16])
    if blockid !=None:
        return blocklist[blockid]
    return blocklist


def showpic(img,landmark=None):
    '''
    展示单张图片，和图片打点，主要用于测试,输入图片和坐标点数组
    :param all_imgs:
    :param all_landmarks:
    :return:
    '''
    point_size = 1
    point_color = (255, 0, 0)  # BGR
    thickness = 4
    tmp_img = copy.deepcopy(img)
    if landmark!=None:
        for idx, point in enumerate(landmark):
            cv2.circle(tmp_img, point, point_size, color=point_color, thickness=thickness)
            #cv2.putText(tmp_img, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 2555, 255))

    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    #cv2.resizeWindow('input_image', 600, 600)
    cv2.imshow('input_image', tmp_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def showedge(img,edge):
    '''
    展示图片脸的边界，测试使用
    :param img:
    :param edge:
    :return:
    '''
    red = (0, 0, 255)  # 8
    cv2.rectangle(img, (edge[3], edge[1]), (edge[2], edge[0]), red, 3)  # 9
    cv2.imshow("Canvas", img)  # 10
    cv2.waitKey(0)  # 11

def landmark_to_label(train_landmarks):
    label_lenghth = len(train_landmarks[0])*2
    return np.array(train_landmarks).reshape(-1, label_lenghth)
def label_to_landmark(label):
    '''
    单个label序列转化成点对
    :param label:
    :return:
    '''
    # print(label.shape)
    points = np.array(label).reshape(-1, 2)
    points = [(int(point[0]), int(point[1])) for point in points]
    return points

def getdata2(number = None):
    '''
    读取进哥给的数据集
    :return:
    '''
    txtpath = "D:\\wangqiang\\source\\lfw_funneled_aligned\\aligned_landmarks2.txt"
    li = os.listdir("D:\\wangqiang\\source\\lfw_funneled_aligned")
    all_imgs =[]
    all_landmarks= []
    index = 0
    with open(txtpath,"r")as r:
        line  = r.readline().strip()

        while line:
            path ="D:/wangqiang/source/lfw_funneled_aligned/"+line
            img = cv2.imread(path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
            points = []
            for i in range(12):
                line = r.readline()
                point = line.strip().split(" ")
                points.append((int(float(point[0])),int(float(point[1]))))
            #showpic(img,points)
            all_imgs.append(img)
            all_landmarks.append(points)
            index+=1
            if number:
                if index>number:
                    break
            line  = r.readline().strip()
    return all_imgs,all_landmarks


if __name__ =="__main__":
    train_imgs, train_landmarks = getdata2(100)
    train_labels = landmark_to_label(train_landmarks)
    train_labels = np.array(train_labels)
    features,values = extract_block_pixdiff(train_imgs,train_labels,100)
    print(len(features))
    print(len(values))
    print(len(values[0]))
