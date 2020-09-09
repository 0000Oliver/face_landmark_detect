import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import copy
from data_preprocess.pic_preprocess import getdata,showpic
def rgb2gray(rgb):
    return np.matmul(rgb,np.array([0.299,0.578,0.114]))

def div(img,cell_x,cell_y,cell_w):
    cell=np.zeros(shape=(cell_x,cell_y,cell_w,cell_w))
    img_x=np.split(img,cell_x,axis=0)
    for i in range(cell_x):
        img_y=np.split(img_x[i],cell_y,axis=1)
        for j in range(cell_y):
            cell[i][j]=img_y[j]
    return cell

def get_bins(grad_cell,ang_cell):
    bins=np.zeros(shape=(grad_cell.shape[0],grad_cell.shape[1],9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn=np.zeros(9)
            grad_list=grad_cell[i,j].flatten()
            ang_list=ang_cell[i,j].flatten()
            left=np.int8(ang_list/20.0)
            right=left+1
            right[right>=8]=0
            left_rit=(ang_list-20*left)/20.0
            right_rit=1.0-left_rit
            binn[left]+=left_rit*grad_list
            binn[right]+=right_rit*grad_list
            bins[i,j]=binn
    return bins


def hog(img,cell_x,cell_y,cell_w):
    if img.ndim==3:
        img=rgb2gray(img)
    gx=cv2.Sobel(img,ddepth=cv2.CV_64F,dx=1,dy=0,ksize=3)
    gy=cv2.Sobel(img,ddepth=cv2.CV_64F,dx=0,dy=1,ksize=3)
    grad=np.sqrt(gx*gx+gy*gy)
    ang=np.arctan2(gx,gy)
    ang[ang<0]=np.pi+ang[ang<0]
    ang*=(180.0/np.pi)
    ang[ang>=180]-=180
    grad_cell=div(grad,cell_x,cell_y,cell_w)
    ang_cell=div(ang,cell_x,cell_y,cell_w)
    bins=get_bins(grad_cell,ang_cell)
    feature=[]
    for i in range(cell_x-1):
        for j in range(cell_y-1):
            tmp=[]
            tmp.append(bins[i,j])
            tmp.append(bins[i+1,j])
            tmp.append(bins[i,j+1])
            tmp.append(bins[i+1,j+1])
            tmp-=np.mean(tmp)
            feature.append(tmp.flatten())
    #plt.imshow(grad,cmap=plt.cm.gray)
    #plt.show()
    return np.array(feature).flatten()


def cut_img_by_landmark(img,landmark,edge=16):
    '''
    输入一张图片，返回按照lardmark点切割的小图片块,edge x是小方块边长的一半，也就是距中心的距离
    :param img:
    :param landmark:
    :return:
    '''
    blocklist = []
    #showpic(img)
    shape = img.shape
    for idx,point in enumerate(landmark):
        block = img[point[1]-edge:point[1]+edge,point[0]-edge:point[0]+edge]

        # if block.shape[0]!=32 or block.shape[1]!=32:
        #     print("图片关键点溢出了边界")
        #     print(idx)
        #     showpic(img,landmark)
        #     print(block)
        #     showpic(block)
        if point[1]-edge<0:
           block = np.vstack((np.zeros((0-(point[1]-edge),2*edge)),block)).astype(np.uint8)
        if point[1]+edge>shape[0]:
            # print(block.shape)
            # print(np.zeros((point[1]+16-shape[0], 32)).shape)
            block = np.vstack((block, np.zeros((point[1]+edge-shape[0], 2*edge)))).astype(np.uint8)
            # print(block)
        if point[0]-edge<0:
           block = np.hstack((np.zeros((2*edge,0-(point[0]-edge))),block)).astype(np.uint8)
        if point[0] + edge > shape[1]:
           block = np.hstack(( block,np.zeros((2*edge, point[0] + edge - shape[1])))).astype(np.uint8)

        blocklist.append(block)
        #showpic(img[point[1]-16:point[1]+16,point[0]-16:point[0]+16])
    return blocklist

def get_hog_feature(img,landmark,block=4,cell_w = 8):
    block_list = cut_img_by_landmark(img,landmark,int(cell_w*block/2))

    cell_x = int(block_list[0].shape[0] / cell_w)
    cell_y = int(block_list[0].shape[1] / cell_w)

    feature = []
    for block in block_list:
        block_feature = hog(block, cell_x, cell_y, cell_w)
        feature.append(block_feature)
    return np.array(feature).flatten()



def landmark_to_label(train_landmarks):
    return np.array(train_landmarks).reshape(-1, 24)
def label_to_landmark(label):
    points = np.array(label).reshape(-1, 2)
    points = [(int(point[0]), int(point[1])) for point in points]
    return points
if __name__ =="__main__":
    SDM_train_path = "D:\\wangqiang\\source\\SMD_dataset\\trainset"
    SDM_test_path = "D:\\wangqiang\\source\\SMD_dataset\\testset"

    train_path = "D:\\wangqiang\\source\\pickeddataset\\trainset"
    test_path = "D:\\wangqiang\\source\\pickeddataset\\testset"

    LEARNING_RATE = 0.1
    TREE_NUMBER = 100

    MAX_DEPTH = 3
    FEATURE_NUMBER = 10000
    train_imgs, train_landmarks = getdata(train_path, 10000)

    face_path = '../data/test.jpg'
    for idx ,img in enumerate(train_imgs):

        get_hog_feature(img,train_landmarks[idx])#(4-1)*(4-1)*9*4*12 =3888
    img = Image.open(face_path)
    img = np.array(img)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    cell_w = 8
    cell_x = int(img.shape[0] / cell_w)
    cell_y = int(img.shape[1] / cell_w)
    feature = hog(img, cell_x, cell_y, cell_w)

