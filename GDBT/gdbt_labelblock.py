import abc
import numpy as np
import logging
from GDBT.decision_tree_labelblock import Decision_tree
from util.picutil import label_to_landmark,showedge,getdata2,landmark_to_label,cut_img_by_landmark,showpic
import copy
import matplotlib.pyplot as plt
import time
import pickle
import datetime
class AbstractBaseGradientBoosting(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def predict(self, data):
        pass

class BaseGradientBoosting(AbstractBaseGradientBoosting):

    def __init__(self,learning_rate, n_trees, max_depth, feature_number,lr_decay = None,edge=30):#lr_decay
        '''
        模型初始化
        :param learning_rate:学习率
        :param n_trees: 树的数量
        :param max_depth: 树的深度
        :param feature_number: 特征数量，每个landmark 截取的block中选的点对数量
        :param lr_decay: 学习率衰减   每过lr_decay个循环  当前学习率减少0.1，如果到0.1 则当前学习率乘以0.1
        :param edge: 每个landmark 截取的block 的边距离中心距离 也就是 边长的一半
        '''
        super().__init__()
        self.learning_rate = learning_rate
        self.learning_rate_cp = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = {}
        self.f_value = {}
        self.feature_number =feature_number
        self.lr_decay =lr_decay
        self.edge = edge
        self.name = 'GDBT_block_pixdiff_{}_{}_{}_{}'.format(self.learning_rate_cp,self.n_trees,self.max_depth,self.feature_number)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='../log/' + self.name + '.log',
                            filemode='w')
        self.logger = logging.getLogger()
        # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')  # ('%(name)-12s: %(levelname)-8s )
        console.setFormatter(formatter)
        self.logger.addHandler(console)

    def fit(self,all_imgs, target,test_imgs= None,test_label=None):
        '''
        训练模型
        :param all_imgs:训练图片
        :param target: 标记点转换的标签值
        :param test_imgs: 测试图片
        :param test_label: 测试label
        :return:
        '''
        train_start_time = datetime.datetime.now()
        self.name += "_{}".format(len(all_imgs))
        self.logger.debug("start training {}:".format(self.name))
        # 初始化 f_0(x)
        # 对于平方损失来说，初始化 f_0(x) 就是 y 的均值
        mean_label = target.mean(axis = 0)
        self.f_value[0]=mean_label
        for i in range(len(all_imgs)-1):
            self.f_value[0]= np.vstack((self.f_value[0],mean_label))
        # 对 m = 1, 2, ..., M

        train_loss_list = []
        test_loss_list = []
        train_loss = self.compute_loss(all_imgs, target)
        self.logger.info("train_loss : {}  ".format(train_loss))
        train_loss_list.append(train_loss)

        if test_imgs != None:
            test_loss = self.compute_loss(test_imgs,test_label)
            self.logger.info("test_loss : {}  ".format(test_loss))
            test_loss_list.append(test_loss)

        for iter in range(1, self.n_trees+1):
            self.logger.info("iter : {}  ".format(iter))
            if self.lr_decay!=None:
                if iter%self.lr_decay==0:
                    if self.learning_rate>=0.2:
                       self.learning_rate-=0.1
                    else:
                        self.learning_rate *=0.1
            self.logger.info("learning rate : {}  ".format( self.learning_rate))
            # 计算负梯度--对于平方误差来说就是残差
            starttime = datetime.datetime.now()
            res_target = target -self.f_value[iter-1]
            self.trees[iter] = Decision_tree(all_imgs=all_imgs,states=self.f_value[iter-1], target=res_target, max_depth=self.max_depth,
                                             logger=self.logger,feature_number=self.feature_number,edge=self.edge).rootnode
            self.f_value[iter]=copy.deepcopy(self.f_value[iter-1])
            for idx , img in enumerate(all_imgs):
                landmark = label_to_landmark(self.f_value[iter][idx])
                blocklist = cut_img_by_landmark(img, landmark, self.edge)
                self.f_value[iter][idx] =self.f_value[iter][idx]+ self.learning_rate*(self.trees[iter].predict(blocklist))
            train_loss = self.compute_loss(all_imgs, target)
            train_loss_list.append(train_loss)
            endtime = datetime.datetime.now()
            self.logger.info("train_loss : {}   costtime : {}".format(train_loss,(endtime-starttime).seconds))

            x_axis = range(len(train_loss_list))
            plt.plot(x_axis, train_loss_list, label='train_loss')  # Plot some data on the (implicit) axes.

            if test_imgs != None:#如果加入测试数据，利用并行计算产生预测误差 不用并行了  发现并行速度更慢
                starttime = datetime.datetime.now()
                test_loss = self.compute_loss(test_imgs, test_label)
                endtime = datetime.datetime.now()
                self.logger.info("test_loss : {}    costtime : {}".format(test_loss,(endtime-starttime).seconds))
                test_loss_list.append(test_loss)
                plt.plot(x_axis, test_loss_list , label='test_loss')  # etc.

            plt_time = datetime.datetime.now()
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.title(self.name + '   cost time :{}'.format((plt_time-train_start_time).seconds))
            plt.legend()
            plt.savefig('../resultpic/'+self.name+'_loss.jpg')
            plt.close('all')
            model_path = '../model/'+self.name+'.txt'
            self.model_save(model_path)
        return self
    def predict(self,img,show=False):
        '''
        利用模型预测图片
        :param img: 图片
        :param show: 是否展示图片
        :return:
        '''

        result = copy.deepcopy(self.f_value[0][0])
        points1 = np.array( result).reshape(-1, 2)
        points1 = [(int(point[0]),int(point[1]))for point in points1]
        if show:
           showpic(img,points1)
        lr =self.learning_rate_cp
        for iter in range(1, len(self.trees)+1):
            if self.lr_decay!=None:
                if iter%self.lr_decay==0:
                    if lr>=0.2:
                       lr-=0.1
                    else:
                        lr *=0.1
            landmark = label_to_landmark(result)
            blocklist = cut_img_by_landmark(img, landmark, self.edge)
            result += lr*self.trees[iter].predict(blocklist)

        points1 = np.array(result).reshape(-1, 2)
        points1 = [(int(point[0]), int(point[1])) for point in points1]
        if show:
           showpic(img, points1)
        return result

    def compute_loss(self,imgs,labels):
        '''
        利用predict计算误差
        :param imgs:
        :param labels:
        :return:
        '''
        loss = 0
        for idx, img in enumerate(imgs):
            pre_label = self.predict(img)
            loss += np.mean(np.square(labels[idx] - pre_label))
        loss = loss / len(imgs)
        return loss

    def model_save(self,path):
        '''
        保存模型
        :param path:
        :return:
        '''

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def model_load(cls,path):
        '''
        加载模型
        :param path:
        :return:
        '''
        with open(path, 'rb')as f:
             GDBT = pickle.load(f)
        return  GDBT


if __name__ =="__main__":

    LEARNING_RATE = 0.5
    TREE_NUMBER =80

    MAX_DEPTH = 3
    FEATURE_NUMBER = 500
    lr_decay = 15
    # train_imgs, train_landmarks = getdata(train_path, 3000)
    # print(np.array(train_landmarks).shape)
    TOTAL_NUMBER = 10000
    train_imgs, train_landmarks = getdata2(TOTAL_NUMBER)
    train_labels = landmark_to_label(train_landmarks)
    train_labels = np.array(train_labels)
    #print(train_labels)
    TRAIN_NUMBER = int(TOTAL_NUMBER*0.8)
    TEST_NUMBER = int(TOTAL_NUMBER*0.2)

    stime = time.time()
    testB = BaseGradientBoosting(LEARNING_RATE,TREE_NUMBER,MAX_DEPTH,FEATURE_NUMBER,lr_decay)
    testB.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    etime = time.time()
    print('costtime: ',etime-stime)



    # for idx ,img in enumerate(train_imgs):
    #     testB.predict(img)

    # testB = BaseGradientBoosting.model_load(model_path)



