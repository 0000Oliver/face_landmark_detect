import cv2
import os
import numpy as np
import math
import random
import copy
from multiprocessing.pool import ThreadPool

import time
def getSource(root_path,save_path,landmark_number):
    '''
    从本地读取图片和landmark点 ，返回处理后的图片和landmark
    :param root_path:
    :return:
    '''
    dirlist = os.listdir(root_path)
    dirlist = [dirname.split(".")[0] for dirname in dirlist]
    codelist = [dirname[:-7] if "mirror" in dirname else dirname for dirname in dirlist]
    codelist = list(set(codelist))
    all_imgs =[]
    all_landmarks = []
    for code in codelist:
        print(code)
        #读取图片和镜像图片
        img = cv2.imread(os.path.join(root_path, code + '.jpg'))#读取图片
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#灰度处理
        img_mirror = cv2.imread(os.path.join(root_path, code + '_mirror.jpg'))
        img_mirror = cv2.cvtColor(img_mirror, cv2.COLOR_BGR2GRAY)

        #print(img.shape)  # 行数 列数 像素数
        #读取坐标点
        with open(os.path.join(root_path, code + '.pts'), 'r')as r:
            lines = r.readlines()
            points = [line.strip().split(' ') for line in lines[3:-1]]
        points = [(int(float(point[0])), int(float(point[1]))) for point in points]
        showpic(img)
        # picked_index = [0,1,8,15,16,17,19,21,22,24,26,33,36,39,42,45,48,54]#原来是68个点 挑选出少的一些点 18
        if landmark_number==12:
            picked_index = [ 8, 17, 21, 22, 26, 33, 36, 39, 42, 45, 48, 54]  # 原来是68个点 挑选出少的一些点12
            points = [points[i] for i in picked_index]

        points_mirror = [(img.shape[1] - point[0], point[1]) for point in points]
        # mirror_index = [4,3,2,1,0,10,9,8,7,6,5,11,15,14,13,12,17,16] #镜像图片的像素点左右位置调整回来18
        if landmark_number == 12:
            mirror_index = [0 , 4 , 3 , 2 , 1 , 5 , 9 , 8 , 7 , 6 , 11 , 10 ]  # 镜像图片的像素点左右位置调整回来12
        else:
            mirror_index = [16, 15, 14 ,13,12,11,10,9,8,7,6,5,4,3,2,1,0,26,25,24,23,22,21,20,19,18,17,27,28,29,30,35,34,33,
                           32,31,45,44,43,42,47,46,39,38,37,36,41,40,54,53,52,51,50,49,48,59,58,57,56,55,64,63,62,61,60,67,66,65]  # 镜像图片的像素点左右位置调整回来68
        points_mirror = [points_mirror[i] for i in mirror_index]

        img,points = img_tureRotate2(img,points,landmark_number)
        img_mirror, points_mirror = img_tureRotate2(img_mirror, points_mirror,landmark_number)
        img,points = img_resize2(img,points,landmark_number)
        img_mirror,points_mirror = img_resize2(img_mirror,points_mirror,landmark_number)
        img,points=img_move_to_center(img,points)
        img_mirror,points_mirror =img_move_to_center(img_mirror,points_mirror)

        path = save_path+"\\"+code
        img,points = img_cut_save(img,points,landmark_number,path)
        path_mirror = save_path+"\\"+code+"_mirror"
        img_mirror, points_mirror = img_cut_save(img_mirror, points_mirror,landmark_number,path_mirror)

        # showpic(img, points)
        # showpic(img_mirror,points_mirror)


        all_imgs.append(img)
        all_imgs.append(img_mirror)
        all_landmarks.append(points)
        all_landmarks.append(points_mirror)

    return all_imgs,all_landmarks

def getAngle(p1,p2):
    '''
    计算两点与x轴的角度，用来旋转，输入两个眼角的坐标点，返回旋转角度
    :param x:
    :param y:
    :return:
    '''
    deta_y = p2[1] - p1[1]
    deta_x = p2[0] - p1[0]
    angle = math.atan2(deta_y, deta_x)#取值在-pi 到pi
    angle = angle  *180/math.pi #转换成角度值
    if angle > 90:
        angle = angle-180
    if angle <-90:
        angle = angle+180
    return angle
def pointRotate(centerpoiont,rpoint,angle):
    '''
    绕着centerpoint 旋转rpoint  返回旋转后的坐标
    :param centerpoiont:
    :param rpoint:
    :param angle:
    :return:
    '''
    angle = -angle*math.pi/180
    cx = centerpoiont[0]
    cy = centerpoiont[1]
    rx = rpoint[0]
    ry = rpoint[1]

    newx = (rx-cx)*math.cos(angle) - (ry-cy)*math.sin(angle) + cx
    newy = (ry-cy)*math.cos(angle) + (rx-cx)*math.sin(angle) + cy
    return (int(newx),int(newy))


def img_tureRotate2(img,landmark,landmark_number):
    '''
    输入图片列表和lardmark列表  旋转图片已经相应的landmark
    :param imgs:
    :param landmarks:
    :return:
    '''
    # 通过眼角的角度旋转图片相应旋转坐标点


    points = landmark

    if landmark_number==12:
        angle = getAngle(points[6], points[9])  # 旋转角度  挑选点后点对位置变化
    else:
        angle = getAngle(points[36], points[45])  # 旋转角度
    center = (int(img.shape[0] / 2), int(img.shape[1] / 2))#图片中心点
    #旋转图片
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    newimg = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    #旋转坐标点
    newpoints =[]
    for i in range(len(points)):
        newpoints.append(pointRotate(center, points[i], angle))

    #showpic(newimg, newpoints)
    return newimg,newpoints

def get_affine_mat(p_ldmk,p_ave_ldmk):
    '''
    求仿射变换矩阵
    :param p_ldmk:
    :param p_ave_ldmk:
    :return:
    '''
    print("test")
    length = len(p_ldmk)
    sumx1 =0
    sumy1 = 0
    sumx2 = 0
    sumy2 = 0
    for i in range(length):
        sumx1 +=p_ldmk[i][0]
        sumy1 +=p_ldmk[i][1]
        sumx2 +=p_ave_ldmk[i][0]
        sumy2 +=p_ave_ldmk[i][1]
    sumx1/=length
    sumy1/=length
    sumx2/=length
    sumy2/=length
    p_s1= []
    p_s2 = []
    for i in range(length):
        p_s1.append([])
        p_s1[i].append(p_ldmk[i][0] - sumx1)
        p_s1[i].append(p_ldmk[i][1] - sumy1)

        p_s2.append([])
        p_s2[i].append(p_ave_ldmk[i][0] - sumx2)
        p_s2[i].append(p_ave_ldmk[i][1] - sumy2)

    temp1=0
    temp2=0
    for i in range(length):
        temp1 += p_s1[i][0] * p_s2[i][0]
        temp1 += p_s1[i][1] * p_s2[i][1]

        temp2 += p_s1[i][0] * p_s1[i][0]
        temp2 += p_s1[i][1] * p_s1[i][1]
    a = temp1/temp2
    temp1 = 0
    for i in range(length):
        temp1+=p_s1[i][0] * p_s2[i][1]
        temp1-=p_s1[i][1] * p_s2[i][0]

    b = temp1 / temp2

    offset_x = (1-a)*sumx1-(-b)*sumy1
    offset_y = (-b)*sumx1+(1-a)*sumy1
    offset_x+=240/2-sumx1
    offset_y+=280/2-sumy1

    p_affine_mat = [a,-b,offset_x,b,a,offset_y]
    return p_affine_mat



# def img_resize()



def img_resize2(img,landmark,landmark_number):
    '''
    把图片和landmark按照眼睛间距平均值缩放，脸部像素大小相同
    :param imgs:
    :param landmarks:
    :return:
    '''

    if landmark_number==12:
        corner_of_eyes = [landmark[6], landmark[9]] # 眼角的位置 挑选点后点对位置变化
    else:
        corner_of_eyes = [landmark[36], landmark[45]]  # 眼角的位置

    eyesdistance = abs(corner_of_eyes[1][0] - corner_of_eyes[0][0])
    avgdistance =90 #调整后的眼睛间距

    points = landmark
    height, width = img.shape[0], img.shape[1]
    # 设置新的图片分辨率框架
    retio =  avgdistance/eyesdistance
    width_new = int(width * retio)
    height_new = int(height * retio)
    newpoints = [(int(point[0]* retio) ,
                  int(point[1]*retio) )for point in points]
    #这里的取整运算可能导致landmark坐标溢出图片大小，进行处理

    newimg = cv2.resize(img, (width_new, height_new))



    return newimg,newpoints



def img_move_to_center(img,landmark):
    '''
    把人脸和landmark移到图片正中心
    :param all_imgs:
    :param landmarks:
    :return:
    '''

    points = landmark
    nppoints = np.array(points)
    facecenter = nppoints.mean(axis=0)
    imgcenter = [img.shape[1]/2,img.shape[0]/2]
    mat_translation = np.float32([[1, 0, imgcenter[0]-facecenter[0]], [0, 1,imgcenter[1]-facecenter[1]]])  # 变换矩阵：设置平移变换所需的计算矩阵：2行3列
    # [[1,0,20],[0,1,50]]   表示平移变换：其中20表示水平方向上的平移距离，50表示竖直方向上的平移距离。
    newimg = cv2.warpAffine(img, mat_translation, (img.shape[1], img.shape[0]))  # 变换函数
    newlandmark = [(int(point[0]+imgcenter[0]-facecenter[0]),int(point[1]+imgcenter[1]-facecenter[1]))for point in points]
    #showpic(img,points)

    return newimg,newlandmark




def img_cut_save(img,landmark,landmark_number,save_path):
    '''
    处理后的图像找到其中的脸部区域，返回长方形的四边，用于下一步取点,截取并保存并保存
    :param all_imgs:
    :param all_landmarks:
    :return:
    '''

    #找到图片中心点，找到图片四边的界限

    points = landmark
    imgcenter = [int(img.shape[1] / 2),int( img.shape[0] / 2)]

    xmargin = 120#图片大小为300*300 则 边框距中心150
    ymargin = 140
    if landmark_number==12:
        chinmargin = imgcenter[1]+ymargin -(landmark[0][1]+30) #下巴经常出到图片外面，特别处理一下,使得下巴点距离图片下沿20个像素点
    else:
        chinmargin = imgcenter[1] + ymargin - (landmark[8][1] + 30)  # 下巴经常出到图片外面，特别处理一下,使得下巴点距离图片下沿20个像素点
    if chinmargin<0:
        imgcenter[1] -=chinmargin


    #求出相对于中心的位置，如果图片过小则补全
    marginx = min(img.shape[1] - (imgcenter[0]+xmargin),imgcenter[0]-xmargin-0)
    marginy = min(img.shape[0] - (imgcenter[1]+ymargin),imgcenter[1]-ymargin-0)
    if marginx < 0:
        border = abs(marginx)
        img = cv2.copyMakeBorder(img, 0, 0, border, border,
                                 cv2.BORDER_CONSTANT, value=[0, 255, 0])
        points = [(point[0] + border, point[1])
                     for point in points]
        imgcenter[0]+=border
    if marginy < 0:
        border = abs(marginy)
        img = cv2.copyMakeBorder(img,border,border,0,0,
                                 cv2.BORDER_CONSTANT, value=[0, 255, 0]
                                 )
        points = [(point[0], point[1] + border)
                  for point in points]
        imgcenter[1]+=border
    edg_of_newimg = [imgcenter[0] - xmargin,
                     imgcenter[0] + xmargin,
                     imgcenter[1] - ymargin,
                     imgcenter[1] + ymargin
                   ]#左边 x极小值，右边 x极大值，下边 y极小值，上边 有极大值

    corners = [(edg_of_newimg[0],edg_of_newimg[2]),
              (edg_of_newimg[0],edg_of_newimg[3]),
              (edg_of_newimg[1],edg_of_newimg[2]),
              (edg_of_newimg[1],edg_of_newimg[3])]
    #根据中心的位置切割图片  调整关键点位置
    #showpic(img, corners)
    newimg = img[ edg_of_newimg[2]:edg_of_newimg[3],
             edg_of_newimg[0]:edg_of_newimg[1]
            ]

    print(newimg.shape)


    newpoints = [(point[0]-int(edg_of_newimg[0]),point[1]-int(edg_of_newimg[2]))
                 for point in points]

    #showpic(img,points)
    #showpic(newimg,newpoints)
    cv2.imwrite(save_path+'.jpg',newimg)
    with open(save_path+'.txt','w') as w:
        for idx ,point in enumerate(newpoints):
            w.write('{},{}'.format(str(point[0]),str(point[1])))
            if idx<67:
              w.write("\n")

    return newimg,newpoints

def getdata(path,img_number):
    '''
    从本地读取处理好的图片和landmark点 ，返回处理后的图片和landmark
    :param root_path:
    :return:
    '''
    dirlist = os.listdir(path)
    codelist = [dirname.split(".")[0] for dirname in dirlist]
    codelist = list(set(codelist))
    all_imgs =[]
    all_landmarks = []
    print(len(codelist))
    for code in codelist[:img_number]:
        #读取图片和镜像图片
        img = cv2.imread(os.path.join(path, code + '.jpg'))#读取图片
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
        #print(img.shape)  # 行数 列数 像素数
        #读取坐标点
        with open(os.path.join(path, code + '.txt'), 'r')as r:
            lines = r.readlines()
            points = [line.strip().split(',') for line in lines]
        points = [(int(float(point[0])), int(float(point[1]))) for point in points]

        # print(code)
        # showpic(img,points)
        # if img.shape[0]!=240 or img.shape[1]!=200:
        #     print(code)
        #     print(img.shape)
        #     showpic(img)
        all_imgs.append(img)
        all_landmarks.append(points)

    return all_imgs,all_landmarks

def getdata2(number = None):
    '''
    读取进哥给的数据集
    :return:
    '''
    txtpath = "/Users/wangqiang/Source/face/aligned_landmarks2.txt"
    # li = os.listdir("D:\\wangqiang\\source\\lfw_funneled_aligned")
    all_imgs =[]
    all_landmarks= []
    index = 0
    with open(txtpath,"r")as r:
        line  = r.readline().strip()

        while line:
            if number:
                if index>=number:
                    break
            path ="/Users/wangqiang/Source/face/"+line
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

            line  = r.readline().strip()
    return all_imgs,all_landmarks

def extractFeature(all_imgs,feature_number):
    '''
    在图片给定的区域随机取点对相减作为特征值，
    :param all_imgs: 图片序列
    :param edgs: 左 右 下 上 四个边界的像素值[161, 825, 87, 744]
    :param feature_number: 要选几个点对
    :return: 返回点对的位置  和点对相减的值
    '''
    features =[]
    values =[]
    shape = all_imgs[0].shape
    for i in range(feature_number):
        deta_x1 =random.randint(0,shape[1]-1)
        deta_y1 = random.randint(0,shape[0]-1)
        deta_x2 = random.randint(0,shape[1]-1)
        deta_y2 = random.randint(0,shape[0]-1)
        features.append([deta_x1,deta_y1,deta_x2,deta_y2])
    for idx ,img in enumerate(all_imgs):
        value =[]
        points =[]
        for feature in features:
            x1 = int(feature[0])
            y1 = int(feature[1])
            x2 = int(feature[2])
            y2 = int(feature[3])
            value.append(int(img[y1][x1])-int(img[y2][x2]))
            points.append((x1,y1))
            points.append((x2, y2))
        # showpic(img,points)
        values.append(value)
    return features,values




def extractFeature2(all_imgs,feature_number):
    '''
    在图片给定的区域随机取点对相减作为特征值，改进使用多线程的方式加速
    :param all_imgs: 图片序列
    :param edgs: 左 右 下 上 四个边界的像素值[161, 825, 87, 744]
    :param feature_number: 要选几个点对
    :return: 返回点对的位置  和点对相减的值
    '''
    thread_feature = 10
    thread_number = int(feature_number/thread_feature)
    def make_feature(thread_feature):
        features = []
        values = []
        shape = all_imgs[0].shape
        for i in range(thread_feature):
            deta_x1 = random.randint(0, shape[1] - 1)
            deta_y1 = random.randint(0, shape[0] - 1)
            deta_x2 = random.randint(0, shape[1] - 1)
            deta_y2 = random.randint(0, shape[0] - 1)
            features.append([deta_x1, deta_y1, deta_x2, deta_y2])
        for idx, img in enumerate(all_imgs):
            value = []
            points = []
            for feature in features:
                x1 = int(feature[0])
                y1 = int(feature[1])
                x2 = int(feature[2])
                y2 = int(feature[3])
                value.append(int(img[y1][x1]) - int(img[y2][x2]))
                points.append((x1, y1))
                points.append((x2, y2))
            # showpic(img,points)
            values.append(value)
        return features,values

    pool = ThreadPool(thread_number)

    returnValues = pool.map(make_feature, [thread_feature for _ in range(thread_number)])
    features ,values = returnValues[0]
    for f,v in returnValues[1:]:
        features = np.vstack((features,f))
        values = np.hstack((values,v))
    return features,values

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
def showpiclist(imgs,landmarks):
    '''
    展示图片序列，测试使用
    :param imgs:
    :param landmarks:
    :return:
    '''
    for i in  range(len(imgs)):
        showpic(imgs[i],landmarks[i])


if __name__ =="__main__":
    # 从文件地址读取图片编号

    train_root = 'D:\\wangqiang\\source\\ibug_300W_large_face_landmark_dataset.tar\\ibug_300W_large_face_landmark_dataset\\helen\\trainset'
    test_root = 'D:\\wangqiang\\source\\ibug_300W_large_face_landmark_dataset.tar\\ibug_300W_large_face_landmark_dataset\\helen\\testset'
    train_root2 = 'D:\\wangqiang\\source\\ibug_300W_large_face_landmark_dataset.tar\\ibug_300W_large_face_landmark_dataset\\lfpw\\trainset'
    test_root2 = 'D:\\wangqiang\\source\\ibug_300W_large_face_landmark_dataset.tar\\ibug_300W_large_face_landmark_dataset\\lfpw\\testset'
    train_root3 = 'D:\\wangqiang\\source\\ibug_300W_large_face_landmark_dataset.tar\\ibug_300W_large_face_landmark_dataset\\afw'
    train_root4 = 'D:\\wangqiang\\source\\ibug_300W_large_face_landmark_dataset.tar\\ibug_300W_large_face_landmark_dataset\\ibug'
    train_path = "D:\\wangqiang\\source\\pickeddataset\\trainset"
    test_path = "D:\\wangqiang\\source\\pickeddataset\\testset"
    train_path2 = "D:\\wangqiang\\source\\pickeddataset\\trainset2"
    test_path2 = "D:\\wangqiang\\source\\pickeddataset\\testset2"
    SDM_train_path = "D:\\wangqiang\\source\\SMD_dataset\\trainset"
    SDM_test_path = "D:\\wangqiang\\source\\SMD_dataset\\testset"
    SDM_train_path2 = "D:\\wangqiang\\source\\SMD_dataset\\trainset2"
    SDM_test_path2 = "D:\\wangqiang\\source\\SMD_dataset\\testset2"
    SDM_train_path68 = "D:\\wangqiang\\source\\SMD_dataset\\trainset68"
    SDM_test_path68 = "D:\\wangqiang\\source\\SMD_dataset\\testset68"
    SDM_train_path3 = "D:\\wangqiang\\source\\SMD_dataset\\trainset3"
    SDM_test_path3= "D:\\wangqiang\\source\\SMD_dataset\\testset3"
    SDM_train_path4 = "D:\\wangqiang\\source\\SMD_dataset\\trainset4"
    SDM_test_path4 = "D:\\wangqiang\\source\\SMD_dataset\\testset4"
    getSource(train_root3,SDM_train_path3,12)
    #train_imgs, train_landmarks = getdata(SDM_test_path, 10000)
    # train_imgs, train_landmarks = getdata(train_path, 1000)





    # all_imgs, all_landmarks =getdata(train_path)
    #
    # features, values = extractFeature(all_imgs, 10)



    #showpiclist(all_imgs,all_landmarks)

