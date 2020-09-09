import numpy as np
import pandas as pd
from data_preprocess.pic_preprocess import *
import logging
import sys
class Node:
    def __init__(self,logger=None,split_feature =None,
                 split_value=None,predict_value = None,depth=None):
        self.split_feature = split_feature
        self.split_value = split_value
        self.predict_value = predict_value
        self.left_child = None
        self.right_child = None
        self.logger = logger
        self.depth = depth

    def predict(self,img):
        # points = np.array( self.predict_value).reshape(-1, 2)
        # points = [(int(point[0]),int(point[1]))for point in points]
        # showpic(img,points)

        if  not self.left_child:
            return self.predict_value
        else:
            x1 = self.split_feature[0]
            y1 = self.split_feature[1]
            x2 = self.split_feature[2]
            y2 = self.split_feature[3]
            f_v = int(img[y1][x1]) - int(img[y2][x2])
            if f_v<self.split_value:
                return self.left_child.predict(img)
            else:
                return  self.right_child.predict(img)


class Decision_tree:
    def __init__(self,all_imgs,target,max_depth,logger,feature_number):
        features, values = extractFeature(all_imgs, feature_number)
        self.data = np.array(values)
        self.target= target
        self.features = np.array(features)
        self.max_depth = max_depth
        self.data_index = range(len(self.data))
        self.nodelist = [Node] * (2**self.max_depth-1)
        self.logger = logger
        self.selected_features =[]
        self.rootnode = self.build_tree(range(len(all_imgs)), 0, 1)


    def calculate_se(self,label_index):
        '''
        利用类中的数据计算目标值与平均数的平方差, 每一个值的平方差之和。 除样本的数量，再除以每个label的参数数量
        :return:
        '''
        label = self.target[label_index, :]
        mean = label.mean(axis = 0)
        difference = np.square(label-mean)/len(label_index)
        d_sum = difference.sum(axis =0)
        return np.mean(d_sum)
    def find_best_fearture(self,data_index):
        '''
        给出总数据中的索引，利用索引匹配类中的数据，计算当前最佳分裂属性和属性值
        :param data_index:
        :return:
        '''
        split_feature =None
        split_value = None
        left_part =None
        right_part =None
        min_Square_difference =sys.maxsize
        tmp_data = self.data[data_index, :]
        for idx ,feature in enumerate(self.features):
            tmp_feature = tmp_data[:,idx]
            for fea_val in range(min(tmp_feature),max(tmp_feature),1):
                left_index = [index for index in data_index if self.data[index,idx]<fea_val]
                right_index =  [index for index in data_index if self.data[index,idx]>=fea_val]
                if len(left_index)==0 or len(right_index)==0:
                    continue
                left_difference = self.calculate_se(left_index)
                right_difference= self.calculate_se(right_index)

                if left_difference+right_difference<min_Square_difference:
                    split_feature = feature
                    split_value = fea_val
                    min_Square_difference = left_difference+right_difference
                    left_part = left_index
                    right_part = right_index

        # print(split_value)
        return split_feature,split_value,left_part,right_part

    def find_best_fearture2(self, data_index):
        '''
        给出总数据中的索引，利用索引匹配类中的数据，计算当前最佳分裂属性和属性值,这个和上一个不同在于用了直方图加速法，速度能提升三倍

        :param data_index:
        :return:
        '''
        split_feature = None
        split_value = None
        left_part = None
        right_part = None
        min_Square_difference = sys.maxsize
        tmp_data = self.data[data_index, :]
        tmp_target = self.target[data_index,:]
        residual = ''
        for idx, feature in enumerate(self.features):

            tmp_feature = tmp_data[:, idx]
            begin = min(tmp_feature)
            end = max(tmp_feature)+1
            bin_number = [0 for _ in range(begin, end, 1)]
            bin_residual = [0 for _ in range(begin, end, 1)]
            for i , val in enumerate(tmp_feature):
                index = val-begin
                bin_number[index]+=1
                bin_residual[index] += np.square(tmp_target[i]).mean()
            total_loss = sum(bin_residual)
            total_number = len(data_index)
            for i in range(1,len(bin_number)-1):
                left_number = sum(bin_number[:i])
                left_residual = sum(bin_residual[:i])
                left_difference = left_residual**2/left_number
                right_difference = (total_loss-left_residual)**2/(total_number-left_number)
                if left_difference + right_difference < min_Square_difference:
                    split_feature = feature
                    split_value = i+begin
                    min_Square_difference = left_difference + right_difference
                    left_part = [index for index in data_index if self.data[index, idx] < split_value]
                    right_part = [index for index in data_index if self.data[index, idx] >= split_value]



        # print(split_value)
        return split_feature, split_value, left_part, right_part


    def node_predict(self,data_index):
        node_label = self.target[data_index]
        return node_label.mean(axis = 0)

    def build_tree(self,data_index,node_number,depth):
           '''
           递归建树，产生并返回节点，将节点存到列表中
           :param data_index:
           :param node_number:
           :param depth:
           :return:
           '''

           if depth <self.max_depth and len(data_index)>1:
               split_feature, split_value, left_part, right_part = self.find_best_fearture2(data_index)
               node = Node(logger=self.logger,split_feature=split_feature,split_value=split_value,predict_value=self.node_predict(data_index),depth=depth)

               node.left_child=self.build_tree(left_part,node_number *2+1,depth+1)
               node.right_child=self.build_tree(right_part,node_number*2+2,depth+1)

               self.nodelist[node_number] = node
               return node
           else:
               node = Node(logger=self.logger,predict_value=self.node_predict(data_index),depth=depth )

               self.nodelist[node_number] = node
               return node







if __name__ =="__main__":
    BATCH = 20
    FEATURE_NUMBER = 10
    MAX_DEPTH = 3

    data_path = '../data/cleaned_data'
    all_imgs1, all_landmarks1 = getdata(data_path)
    label = np.array(all_landmarks1).reshape(-1,136)

    #print(all_imgs1[0])
    logger = logging.getLogger('log')
    testtree = Decision_tree(all_imgs=all_imgs1, target=label, max_depth=MAX_DEPTH, logger=logger)
    node = testtree.rootnode
    label =node.predict(all_imgs1[0])
    # testtree.build_tree(range(BATCH),0,1)





