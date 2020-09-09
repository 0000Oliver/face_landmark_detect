from data_preprocess.pic_preprocess import getdata,showpic,getdata2
from SDM.Hog_feature import get_hog_feature,landmark_to_label,label_to_landmark
import  numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import pandas as pd
import datetime
from sklearn import decomposition
import logging


class DimensionValueError(ValueError):
    """定义异常类"""
    pass


class PCA(object):
    """定义PCA类"""

    def __init__(self, x, n_components=None):
        self.x = x
        self.dimension = x.shape[1]

        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")

        self.n_components = n_components

    def cov(self):
        """求x的协方差矩阵"""
        x_T = np.transpose(self.x)                           #矩阵转置
        x_cov = np.cov(x_T)                                  #协方差矩阵
        return x_cov

    def get_feature(self):
        """求协方差矩阵C的特征值和特征向量"""
        x_cov = self.cov()
        a, b = np.linalg.eig(x_cov)
        m = a.shape[0]
        c = np.hstack((a.reshape((m,1)), b))
        c_df = pd.DataFrame(c)
        c_df_sort = c_df.sort_values(by=0, ascending=False)
        return c_df_sort

    def explained_varience_(self):
        c_df_sort = self.get_feature()
        return c_df_sort.values[:, 0]

    def paint_varience_(self):
        explained_variance_ = self.explained_varience_()
        plt.figure()
        plt.plot(explained_variance_, 'k')
        plt.xlabel('n_components', fontsize=16)
        plt.ylabel('explained_variance_', fontsize=16)
        plt.show()

    def reduce_dimension(self):
        """指定维度降维和根据方差贡献率自动降维"""
        c_df_sort = self.get_feature()
        varience = self.explained_varience_()

        if self.n_components:  # 指定降维维度
            p = np.transpose(c_df_sort.values[0:self.n_components, 1:])
            y = np.dot(self.x, p)
            return y, p

        varience_sum = sum(varience)
        varience_radio = varience / varience_sum

        varience_contribution = 0
        for R in range(self.dimension):
            varience_contribution += varience_radio[R]
            if varience_contribution >= 0.99:
                break

        p = np.transpose(c_df_sort.values[0:R + 1, 1:])  # 取前R个特征向量
        y = np.dot(self.x, p)
        return y, p

class GSDM_model():#不用每一步都保存pca  base  只保存一个层  pcabase
    def __init__(self,leaning_rate,feature_block,blocksize =8):
        self.learning_rate = leaning_rate
        self.feature_block = feature_block
        self.block_size = blocksize
        self.f_value={}
        self.derection=[]
        self.loss = 100
        self.pca_base= None
        self.name= 'GSDM4_{}_{}'.format(self.learning_rate,self.feature_block)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='../log/'+self.name+'.log',
                            filemode='w')
        self.logger = logging.getLogger()
        # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')  # ('%(name)-12s: %(levelname)-8s )
        console.setFormatter(formatter)
        # logging.getLogger('').addHandler(console)
        self.logger.addHandler(console)

    def get_feature(self,img,state):
        '''
        利用图片和当前关键点计算特征，多次使用并且 直接用了类的参数所以 创建一个类方便修改
        :param img:
        :param state:
        :return:
        '''

        return get_hog_feature(img, label_to_landmark(state), self.feature_block,self.block_size)# (4-1)*(4-1)*9*4*12 =3888

    def get_phis(self,imgs,label):
        phi_star = [self.get_feature(imgs[idx], label[idx]) for idx in
                    range(len(imgs))] # (4-1)*(4-1)*9*4*12 =3888
        phi_star = np.array(phi_star)
        return phi_star


    def comput_detaX_phi(self,imgs, state, label,phi_star):
        deta_X = label - state
        phi = self.get_phis(imgs,state)
        deta_phi = phi_star-phi
        return deta_X,deta_phi

    def devide_subset(self,deta_X,deta_phi):
        '''
        利用pca算法求出 deta_X,deta_phi 主要方向  利用主要方向将数据集分成8个子集
        :param deta_X:
        :param deta_phi:
        :return:
        '''
        pca = PCA(deta_X,2)
        pca_deta_X ,pac_base_X = pca.reduce_dimension()
        pca = PCA(deta_phi,1)
        pca_deta_phi,pac_base_phi = pca.reduce_dimension()
        subset = [[]for i in range(8)]
        for idx in range(len(deta_X)):
            pca_x = pca_deta_X[idx]
            pca_phi = pca_deta_phi[idx]
            if pca_x[0]>=0:
                if pca_x[1]>=0:
                    if pca_phi[0]>=0:
                        subset[0].append(idx)
                    else:
                        subset[1].append(idx)
                else:
                    if pca_phi[0]>=0:
                        subset[2].append(idx)
                    else:
                        subset[3].append(idx)
            else:
                if pca_x[1]>=0:
                    if pca_phi[0]>=0:
                        subset[4].append(idx)
                    else:
                        subset[5].append(idx)
                else:
                    if pca_phi[0]>=0:
                        subset[6].append(idx)
                    else:
                        subset[7].append(idx)
        return subset,(pac_base_X,pac_base_phi)


    def Descent(self, imgs, state, label):
        '''
        用最小二乘法求出y=ax+b 中的ab
        :param imgs:
        :param state:
        :param label:
        :return:
        '''
        delta_X = label - state

        phis = self.get_phis(imgs,state)
        phi_one = np.ones(phis.shape)
        A = np.hstack((phis, phi_one))
        A = np.mat(A)
        X = np.dot(np.dot(np.linalg.pinv(np.dot(A.T, A)), A.T), delta_X)
        return X, np.dot(A, X)


    def  fit(self,imgs,labels,test_imgs = None,test_labels=None):
        self.logger.info("GSDM learning rate : {}   number of featureblock   {}   number of picture : {}".format(self.learning_rate,self.feature_block,len(imgs)))
        #print("GSDM learning rate : {}   number of featureblock   {}   number of picture : {}".format(self.learning_rate,self.feature_block,len(imgs)))
        imgs = np.array(imgs)
        state = []
        for i in range(len(labels)):
            state.append(np.array(labels).mean(axis=0))
        state = np.array(state)
        self.f_value[0] = state
        phi_star = self.get_phis(imgs, labels)

        train_loss_list = []
        test_loss_list = []
        train_loss = self.compute_loss(state, labels)
        train_loss_list.append(train_loss)
        self.logger.info("train_loss : {}".format(train_loss) )
        #print()
        if test_imgs != None:
            test_loss = 0
            for idx, img in enumerate(test_imgs):
                state = self.predict(img,test_labels[idx])
                test_loss += self.compute_loss(state, test_labels[idx])
            test_loss = test_loss / len(test_imgs)
            test_loss_list.append(test_loss)
            self.logger.info("test_loss : {}".format(test_loss) )
            # print()

        self.loss = train_loss
        iter = 0
        deta_X, deta_phi = self.comput_detaX_phi(imgs, self.f_value[0], labels, phi_star)
        subsets, pca_base = self.devide_subset(deta_X, deta_phi)
        self.pca_base = pca_base
        while self.loss>0.1:
            self.logger.info("epoch : {}".format(iter))
            #print()
            starttime = datetime.datetime.now()
            self.f_value[iter + 1] = copy.deepcopy(self.f_value[iter])
            self.derection.append({})
            for idx,setindex in enumerate(subsets):
                subimgs = imgs[setindex]
                substate = self.f_value[iter][setindex]
                sublabel = labels[setindex]
                self.derection[iter][idx],strid = self.Descent(subimgs,substate,sublabel)
                self.f_value[iter + 1][setindex]+=self.learning_rate* strid
                # for i,index in enumerate(setindex):
                #    print(self.f_value[iter+1][index].shape)
                #    print(strid[i].reshape(24).shape)
                #    self.f_value[iter+1][index] += self.learning_rate* strid[i].reshape(-1)
            train_loss = self.compute_loss(self.f_value[iter+1],labels)
            train_loss_list.append(train_loss)
            self.loss = train_loss
            endtime = datetime.datetime.now()
            self.logger.info("train_loss : {}   costtime : {}".format(self.loss,(endtime-starttime).seconds))
            #print()
            # print("train_loss : ", self.loss)
            if test_imgs != None:
                starttime = datetime.datetime.now()
                test_loss = 0
                for idx ,img in enumerate(test_imgs):
                    state = self.predict(img,test_labels[idx])
                    test_loss+= self.compute_loss(state,test_labels[idx])
                test_loss = test_loss/len(test_imgs)
                if test_loss>test_loss_list[-1]:
                    self.logger.info("收敛完成")
                    break
                test_loss_list.append(test_loss)
                endtime = datetime.datetime.now()
                self.logger.info("test_loss : {}   costtime : {}".format(test_loss, (endtime - starttime).seconds))



            iter+=1
            x_axis = range(len(train_loss_list))
            plt.plot(x_axis, train_loss_list, label='train_loss')  # Plot some data on the (implicit) axes.
            if test_imgs != None: plt.plot(x_axis, test_loss_list, label='test_loss')  # etc.
            plt.xlabel('iter')
            plt.ylabel('loss')
            figpath = '../resultpic/'+self.name+'_{}_loss.jpg'.format(len(imgs))
            if test_imgs != None:figpath = '../resultpic/'+self.name+'_{}_loss_withtest.jpg'.format(len(imgs))
            plt.title(figpath.split('/')[-1])
            plt.legend()
            plt.savefig(figpath)
            plt.close('all')
            model_path = '../model/'+self.name+'_{}_.txt'.format(len(imgs))

            self.model_save(model_path)

    def predict(self, img, pre_label, show=False):

        state = [copy.deepcopy(self.f_value[0][0])]
        if show:
            showpic(img, label_to_landmark(state))
        phi_star = self.get_feature(img, pre_label)
        for i in range(len(self.derection)):

            deta_X = pre_label - state
            deta_phi = phi_star - self.get_feature(img, state)
            pca_deta_X = np.asarray(np.dot(deta_X, self.pca_base[0]))
            # print(deta_X.shape)#(1, 24)
            # print(self.pca_base[i][0].shape)#(24, 7)
            # print(pca_deta_X.shape)#(1, 7)
            pca_deta_phi = np.dot(deta_phi, self.pca_base[1])
            # print(deta_phi.shape)#(1728,)
            # print(self.pca_base[i][1].shape)#(1728, 88)
            # print(pca_deta_phi.shape)#(88,)
            if pca_deta_X[0][0] >= 0:
                if pca_deta_X[0][1] >= 0:
                    if pca_deta_phi[0] >= 0:
                        subindex = 0
                    else:
                        subindex = 1
                else:
                    if pca_deta_phi[0] >= 0:
                        subindex = 2
                    else:
                        subindex = 3
            else:
                if pca_deta_X[0][1] >= 0:
                    if pca_deta_phi[0] >= 0:
                        subindex = 4
                    else:
                        subindex = 5
                else:
                    if pca_deta_phi[0] >= 0:
                        subindex = 6
                    else:
                        subindex = 7
            state += self.learning_rate * self.compute_strid(img, state, self.derection[i][subindex])
        if show:
            showpic(img, label_to_landmark(state))
        return state
    def compute_strid(self,img, state, decent_map):
        '''
                输入单张图片 和 标注 利用计算出来的每一步方向求出每一步的步伐
                :param imgs:
                :param state:
                :param decent_map:
                :return:
                '''
        phi = self.get_feature(img, state)  # (4-1)*(4-1)*9*4*12 =3888
        phi = np.array(phi)
        phi_one = np.ones(phi.shape)
        A = np.hstack((phi, phi_one))
        A = np.mat(A)
        return np.dot(A, decent_map)


    def compute_loss(self,state, label):
        '''
        计算当前位置和目标位置的误差
        :param state:
        :param label:
        :return:
        '''
        loss = state - label
        return np.square(loss).mean()

    def model_save(self, path):

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def model_load(cls, path):
        with open(path, 'rb')as f:
            SDM = pickle.load(f)
        return SDM


if __name__ =="__main__":
    SDM_train_path = "D:\\wangqiang\\source\\SMD_dataset\\alltrainset"
    SDM_test_path = "D:\\wangqiang\\source\\SMD_dataset\\alltestset"

    LEARNING_RATE = 0.1
    FEATRURE_BLOCK  =4
    TRAIN_NUMBER = 9000
    TEST_NUMBER = 2000

    # train_imgs, train_landmarks = getdata(SDM_train_path, 6200)
    # train_labels = landmark_to_label(train_landmarks)
    # train_labels = np.array(train_labels)
    # test_imgs, test_landmarks = getdata(SDM_test_path, 1000)
    # test_labels = landmark_to_label(test_landmarks)
    # test_labels = np.array(test_labels)
    #
    # test_GSDM = GSDM_model(LEARNING_RATE)
    # test_GSDM.fit(train_imgs, train_labels, test_imgs, test_labels)
    train_imgs, train_landmarks = getdata2()
    train_labels = landmark_to_label(train_landmarks)
    train_labels = np.array(train_labels)

    print(len(train_imgs))
    # test_GSDM = GSDM_model(LEARNING_RATE,5)
    # test_GSDM.fit(train_imgs[:5000],train_labels[:5000],train_imgs[-2000:],train_labels[-2000:])
    # test_GSDM = GSDM_model(LEARNING_RATE,5)
    # test_GSDM.fit(train_imgs[:7000],train_labels[:7000],train_imgs[-2000:],train_labels[-2000:])
    test_GSDM = GSDM_model(LEARNING_RATE,FEATRURE_BLOCK)
    test_GSDM.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    # test_GSDM = GSDM_model(LEARNING_RATE,5)
    # test_GSDM.fit(train_imgs[:9000],train_labels[:9000],train_imgs[-3000:],train_labels[-3000:])
    # test_GSDM = GSDM_model(LEARNING_RATE,6)
    # test_GSDM.fit(train_imgs[:9000],train_labels[:9000],train_imgs[-3000:],train_labels[-3000:])

    print('训练完成：')
    test_x = train_imgs[-10:]
    test_y= train_labels[-10:]
    for idx  in range(10):
        test_GSDM.predict(test_x[idx],test_y[idx],True)