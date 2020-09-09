import numpy as np
from util.picutil import label_to_landmark,landmark_to_label,getdata2,get_block_pixdiff,cut_img_by_landmark,extract_single_pixdiff
import sys
import datetime
import operator
from functools import reduce


class Node:
    def __init__(self,logger=None,split_feature =None,
                 split_value=None,predict_value = None,depth=None):
        '''

        :param logger:日志
        :param split_feature:分裂特征  这里具体为pixdiff 的点对信息    [blockid,x1,y1,x2,y2]
        :param split_value: 分列特征的分裂阈值
        :param predict_value: 节点预测值 只有叶子节点有
        :param depth: 节点的深度
        '''
        self.split_feature = split_feature
        self.split_value = split_value
        self.predict_value = predict_value
        self.left_child = None
        self.right_child = None
        self.logger = logger
        self.depth = depth


    def predict(self,blocklist):
        '''
        节点便利预测，利用输入的图片提取特征，
        :param blocklist: 每个树之前利用当前状态截取的landmark周围的block列表
        :return:
        '''
        if  not self.left_child:
            return self.predict_value
        else:
            f_v = extract_single_pixdiff(blocklist,self.split_feature)#从图片中提取单个特征的值
            if f_v<self.split_value:
                return self.left_child.predict(blocklist)
            else:
                return  self.right_child.predict(blocklist)


class Decision_tree:#_block_label:
    def __init__(self,all_imgs,states,target,max_depth,logger,feature_number,edge=30):
        '''

        :param all_imgs:所有图片
        :param states: 当前点位置
        :param target: 点位置的残差
        :param max_depth:树的深度
        :param logger:日志
        :param feature_number:每个block中选的点对数量
        :param edge:block大小 边长的一半
        '''

        features, values = get_block_pixdiff(all_imgs, states,feature_number,edge)#利用图片和 当前label值分割block提取特征
        self.data = np.array(values)
        self.target= target
        self.states =states
        self.features = np.array(features)
        self.max_depth = max_depth
        self.data_index = range(len(self.data))#数据的索引 后续数据处理都使用索引
        self.logger = logger
        self.selected_features =[]
        self.rootnode = self.build_tree(range(len(all_imgs)), 0, 1)
        self.edge =edge


    def calculate_se(self,label_index):
        '''
        计算这一部分数据的均方差的和   注意有坑：不要除数据的个数 直接求和
        :param label_index:当前节点的数据的索引
        :return: 返回均方差
        '''

        label = self.target[label_index, :]
        mean = label.mean(axis = 0)
        difference = np.square(label-mean)
        d_sum = difference.sum(axis =0)

        return np.mean(d_sum)

    def find_best_fearture(self,data_index):
        '''
        给出总数据中的索引，利用索引匹配类中的数据，计算当前最佳分裂属性和属性值
        :param data_index:
        :return:返回最佳分类节点 分类阈值  分裂两部分的索引
        '''
        data_index = np.array(data_index)
        starttime = datetime.datetime.now()
        split_feature =None
        split_value = None
        left_part =None
        right_part =None
        min_Square_difference =sys.maxsize
        tmp_data = self.data[data_index, :]
        self.logger.debug("find_best_fearture   data shape:{}".format(tmp_data.shape))
        for idx ,feature in enumerate(self.features):
            tmp_feature = tmp_data[:,idx]

            for fea_val in range(min(tmp_feature),max(tmp_feature),1):
                left_index = data_index[np.where(self.data[data_index, idx] < fea_val)]
                right_index = data_index[np.where(self.data[data_index, idx] >= fea_val)]

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
        endtime = datetime.datetime.now()
        self.logger.debug("split_feature: {}  split_value  :{}   costtime : {}".format(split_feature,split_value,(endtime-starttime).seconds))
        return split_feature,split_value,left_part,right_part

    def find_best_fearture2(self, data_index):
        '''
        给出总数据中的索引，利用索引匹配类中的数据，计算当前最佳分裂属性和属性值,这个和上一个不同在于用了直方图加速法，速度能提升至少三倍
        :param data_index:
        :return:返回最佳分类节点 分类阈值  分裂两部分的索引
        '''

        starttime = datetime.datetime.now()
        split_feature = None
        split_value = None
        left_part = None
        right_part = None
        data_index = np.array(data_index)
        min_Square_difference = sys.maxsize
        tmp_data = self.data[data_index, :]
        tmp_target = self.target[data_index,:]
        self.logger.debug("find_best_fearture2   data shape:{}".format(tmp_data.shape))

        for idx, feature in enumerate(self.features):

            tmp_feature = tmp_data[:, idx]
            begin = min(tmp_feature)
            end = max(tmp_feature)+1
            bin_number = [0 for _ in range(begin, end, 1)]
            bin_residual = [np.zeros(tmp_target[0].shape) for _ in range(begin, end, 1)]#这里有坑  之前的计算都是先对label求和在进行  平方差的运算  其实应该先对每个label求平方差 再求和
            bin_residual = np.array(bin_residual)
            for i , val in enumerate(tmp_feature):
                index = val-begin
                bin_number[index]+=1
                bin_residual[index] = bin_residual[index]+tmp_target[i]#(sum(tmp_target[i]))#.mean()#np.square(tmp_target[i]).mean()
            total_residual = bin_residual.sum(axis = 0)
            total_number = len(data_index)
            for i in range(1,len(bin_number)-1):
                left_number = sum(bin_number[:i])
                right_number = total_number-left_number
                left_residual = bin_residual[:i].sum(axis = 0)
                right_residual = total_residual-left_residual
                left_difference = -sum(np.square(left_residual))/left_number
                right_difference = -sum(np.square(right_residual))/right_number
                if left_difference + right_difference < min_Square_difference:
                    split_feature = feature
                    split_value = i+begin
                    min_Square_difference = left_difference + right_difference

                    left_part = data_index[np.where(self.data[data_index, idx] < split_value)]
                    right_part = data_index[np.where(self.data[data_index, idx] >= split_value)]
        endtime = datetime.datetime.now()
        self.logger.debug("split_feature: {}  split_value  :{}   costtime : {}".format(split_feature, split_value,
                                                                                       (endtime - starttime).seconds))

        return split_feature, split_value, left_part, right_part

    def node_predict(self,data_index):
        '''
        求节点中的平均值，作为节点的预测值
        :param data_index:
        :return:
        '''
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
           self.logger.debug("build_tree---node_number: {}  depth  :{}".format(node_number, depth,))
           if depth <self.max_depth and len(data_index)>1:
               split_feature, split_value, left_part, right_part = self.find_best_fearture2(data_index)

               node = Node(logger=self.logger,split_feature=split_feature,split_value=split_value,depth=depth)

               node.left_child=self.build_tree(left_part,node_number *2+1,depth+1)
               node.right_child=self.build_tree(right_part,node_number*2+2,depth+1)

               return node
           else:
               node = Node(logger=self.logger,predict_value=self.node_predict(data_index),depth=depth )

               return node

    # def predict(self,laststate,img):
    #     '''
    #     树的预测方法  目前没用  直接使用节点预测
    #     :param laststate:
    #     :param img:
    #     :return:
    #     '''
    #     landmark = label_to_landmark(laststate)
    #     blocklist = cut_img_by_landmark(img, landmark, self.edge)
    #     result = self.rootnode.predict(blocklist)
    #     return result







if __name__ =="__main__":
    BATCH = 20
    FEATURE_NUMBER = 10
    MAX_DEPTH = 3

    train_imgs, train_landmarks = getdata2(100)
    train_labels = landmark_to_label(train_landmarks)
    train_labels = np.array(train_labels)
    print(train_labels)



    #print(all_imgs1[0])
    # logger = logging.getLogger('log')
    # testtree = Decision_tree(all_imgs=all_imgs1, target=label, max_depth=MAX_DEPTH, logger=logger)
    # node = testtree.rootnode
    # label =node.predict(all_imgs1[0])
    # testtree.build_tree(range(BATCH),0,1)





