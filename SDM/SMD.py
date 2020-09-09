from data_preprocess.pic_preprocess import getdata,showpic,getdata2
from SDM.Hog_feature import get_hog_feature,landmark_to_label,label_to_landmark
import  numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import datetime
import logging
from util.myPCA import PCA


class SDM_model():
    def __init__(self,leaning_rate,feature_block=4,blocksize =8,pca_number = None,lr_decay=None,max_iter =None):
        self.learning_rate = leaning_rate
        self.feature_block = feature_block#一个特征点的框边上切割几次 平方个小框
        self.block_size = blocksize#小框的边长
        self.f_value={}
        self.derection={}
        self.loss = 100
        self.PCA_number = pca_number
        self.pca_base =  []
        self.lr_decay = lr_decay
        self.max_iter = max_iter
        self.name = 'SDM_{}_{}_{}_{}_{}'.format(self.learning_rate, self.feature_block,self.block_size,self.PCA_number,self.lr_decay
                                             )

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            filename='../log/tmp.log',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                             filemode='w')
        self.logger = logging.getLogger(self.name)
        # 定义一个FileHandler，将INFO级别或更高的日志信息记录到log文件，并将其添加到当前的日志处理对象#
        fh = logging.FileHandler('../log/' + self.name + '.log', mode='w')
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')  # ('%(name)-12s: %(levelname)-8s )
        console.setFormatter(formatter)

        self.logger.addHandler(console)
    def get_feature(self,img,state):
        '''
        利用图片和当前关键点计算特征，多次使用并且 直接用了类的参数所以 创建一个类方便修改
        :param img:
        :param state:
        :return:
        '''

        return get_hog_feature(img, label_to_landmark(state), self.feature_block,self.block_size)# (4-1)*(4-1)*9*4*12 =3888

    def Descent(self,imgs, state, label):
        '''
        用最小二乘法求出y=ax+b 中的ab
        :param imgs:
        :param state:
        :param label:
        :return:
        '''
        delta_X = label - state
        phi = [self.get_feature(imgs[idx], state[idx]) for idx in#
               range(len(imgs))]  # (4-1)*(4-1)*9*4*12 =3888
        phi = np.array(phi)
        if self.PCA_number != None:
            starttime = datetime.datetime.now()
            pca = PCA(phi,self.PCA_number)
            phi ,pca_base = pca.reduce_dimension()
            self.pca_base.append(pca_base)
            endtime = datetime.datetime.now()
            self.logger.debug("PCA shape : {} cost time: {}".format(str(phi.shape),(endtime-starttime).seconds))

        phi_one = np.ones(phi.shape)
        A = np.hstack((phi, phi_one))
        A = np.mat(A)
        X = np.dot(np.dot(np.linalg.pinv(np.dot(A.T, A)), A.T), delta_X)
        return X, np.dot(A, X)
    def fit(self,imgs,labels,test_imgs = None,test_labels=None):
        fit_time = datetime.datetime.now()
        self.name += "_{}".format(len(imgs))
        self.logger.info(self.name+"  learning rate : {}  feature block :  {}  block size : {}  PCA number : {}  lr_decay : {} number of picture : {}".
                         format(self.learning_rate, self.feature_block,self.block_size,self.PCA_number,self.lr_decay,len(imgs)))
        state = []
        for i in range(len(labels)):
            state.append(np.array(labels).mean(axis=0))
        state = np.array(state)

        self.f_value[0] = state
        train_loss_list = []
        test_loss_list = []
        train_loss = self.compute_loss(imgs, labels)
        train_loss_list.append(train_loss)
        self.logger.info("train_loss : {}".format(train_loss) )
        if test_imgs != None:

            test_loss = self.compute_loss(test_imgs, test_labels)
            test_loss_list.append(test_loss)
            self.logger.info("test_loss : {}".format(test_loss) )


        self.loss = train_loss
        iter = 0
        lr = self.learning_rate
        while self.loss>0.1:
            self.logger.info("epoch : {}   lr :  {}".format(iter,lr))
            if self.lr_decay!=None:
                if (iter+1)%self.lr_decay ==0:
                    if lr >= 0.2:
                        lr -= 0.1
                    else:
                        lr *= 0.1

            starttime = datetime.datetime.now()
            self.derection[iter],strid = self.Descent(imgs,self.f_value[iter],labels)
            self.f_value[iter+1] = self.f_value[iter] + lr * strid
            # train_loss = self.compute_loss(self.f_value[iter+1],labels)
            train_loss = self.compute_diff(self.f_value[iter+1], labels)
            train_loss_list.append(train_loss)
            self.loss = train_loss
            endtime = datetime.datetime.now()
            self.logger.info("train_loss : {}   costtime : {}".format(self.loss,(endtime-starttime).seconds))
            if test_imgs != None:
                starttime = datetime.datetime.now()
                test_loss = self.compute_loss(test_imgs,test_labels)
                test_loss_list.append(test_loss)
                endtime = datetime.datetime.now()
                self.logger.info("batch_test_loss : {}   costtime : {}".format(test_loss, (endtime - starttime).seconds))
                if test_loss>test_loss_list[-2]-0.02:
                    self.logger.info("收敛完成")
                    break
            iter+=1


            x_axis = range(len(train_loss_list))
            plt.plot(x_axis, train_loss_list, label='train_loss')  # Plot some data on the (implicit) axes.
            if test_imgs != None: plt.plot(x_axis, test_loss_list, label='test_loss')  # etc.
            plt.xlabel('iter')
            plt.ylabel('loss')
            figpath = '../resultpic/'+self.name+'.jpg'
            if test_imgs != None:figpath = '../resultpic/'+self.name+'_withtest.jpg'
            plt_time = datetime.datetime.now()
            plt.title(self.name+"  cost time :{}".format((plt_time-fit_time).seconds))
            plt.legend()
            plt.savefig(figpath)
            plt.close('all')
            model_path = '../model/'+self.name+'.txt'

            self.model_save(model_path)
            if self.max_iter!=None:
                if iter >=self.max_iter:
                    break
        self.logger.removeHandler(self.logger.handlers[-1])
        self.logger.removeHandler(self.logger.handlers[-1])


    def compute_strid(self,img, state, decent_map_idx):
        '''
        输入单张图片 和 标注 利用计算出来的每一步方向求出每一步的步伐
        :param imgs:
        :param state:
        :param decent_map:
        :return:
        '''
        decent_map = self.derection[decent_map_idx]
        phi = self.get_feature(img, state)  # (4-1)*(4-1)*9*4*12 =3888
        phi = np.array(phi)
        if self.PCA_number != None:
            #pca = PCA(phi, self.PCA_number)
            phi = np.dot(phi, self.pca_base[decent_map_idx])

        phi_one = np.ones(phi.shape)
        A = np.hstack((phi, phi_one))
        A = np.mat(A)
        return np.dot(A, decent_map)

    def predict(self,img,show = None):
        state = [copy.deepcopy(self.f_value[0][0])]
        if show:
            showpic(img, label_to_landmark(self.f_value[0][0]))
        lr = self.learning_rate
        for i in range(len(self.derection)):
            if self.lr_decay != None:
                if (i+1)%self.lr_decay ==0:
                    if lr >= 0.2:
                        lr -= 0.1
                    else:
                        lr *= 0.1

            state+=lr*self.compute_strid(img,state,i)
        if show:
            showpic(img, label_to_landmark(state))
        return state
    def predict_test(self,img):
        print("预测")
        showpic(img,label_to_landmark(self.f_value[0][0]))
        state = [copy.deepcopy(self.f_value[0][0])]
        lr = self.learning_rate

        for i in range(len(self.derection)):
            if self.lr_decay != None:
                if (i + 1) % self.lr_decay == 0:
                    if lr >= 0.2:
                        lr -= 0.1
                    else:
                        lr *= 0.1
            state+=lr*self.compute_strid(img,state,i)
        showpic(img, label_to_landmark(state))
        return state
    def compute_strid_batch(self,imgs, states, decent_map_idx):
        '''
        输入多张图片 和 标注 利用计算出来的每一步方向求出每一步的步伐
        :param imgs:
        :param state:
        :param decent_map:
        :return:
        '''
        decent_map = self.derection[decent_map_idx]
        phi = [self.get_feature(imgs[idx], states[idx])for idx in range(len(imgs))]  # (4-1)*(4-1)*9*4*12 =3888
        phi = np.array(phi)
        if self.PCA_number != None:
            #pca = PCA(phi, self.PCA_number)
            phi = np.dot(phi, self.pca_base[decent_map_idx])
            #self.pca_base.append(pca_base)
        phi_one = np.ones(phi.shape)
        A = np.hstack((phi, phi_one))
        A = np.mat(A)
        return np.dot(A, decent_map)

    def predict_batch(self,imgs):
        state = [copy.deepcopy(self.f_value[0][0]) for _ in range(len(imgs))]
        lr = self.learning_rate
        for i in range(len(self.derection)):
            if self.lr_decay != None:
                if (i + 1) % self.lr_decay == 0:
                    if lr >= 0.2:
                        lr -= 0.1
                    else:
                        lr *= 0.1
            state+=lr*self.compute_strid_batch(imgs,state,i)
        return state

    def compute_diff(self,tmp_label,label):
        '''
        计算两个label位置的平方误差
        :param tmp_label:
        :param label:
        :return:
        '''
        loss = np.array(tmp_label - label)
        result =np.square(loss).mean()
        #print(type(result))
        return result
    def compute_loss(self,test_imgs, label):
        '''
        计算当前预测值和目标位置的误差
        :param state:
        :param label:
        :return:
        '''
        tmp_label = np.array(self.predict_batch(test_imgs))
        return self.compute_diff(tmp_label,label)

    def model_save(self,path):

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def model_load(cls,path):
        with open(path, 'rb')as f:
             SDM = pickle.load(f)
        return  SDM






if __name__ =="__main__":

    LEARNING_RATE = 0.1
    TOTAL_NUMBER = 100
    TRAIN_NUMBER = int(TOTAL_NUMBER*0.8)
    TEST_NUMBER = int(TOTAL_NUMBER*0.2)
    train_imgs, train_landmarks = getdata2(TOTAL_NUMBER)
    train_labels = landmark_to_label(train_landmarks)
    train_labels = np.array(train_labels)
    print(len(train_imgs))

    test_SMD = SDM_model(LEARNING_RATE,pca_number=100)
    test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    test_SMD = SDM_model(LEARNING_RATE,pca_number=200)
    test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    # test_SMD = SDM_model(LEARNING_RATE,pca_number=400)
    # test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    # test_SMD = SDM_model(LEARNING_RATE,pca_number=500)
    # test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    # test_SMD = SDM_model(LEARNING_RATE,pca_number=600)
    # test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    # test_SMD = SDM_model(LEARNING_RATE,pca_number=700)
    # test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    # test_SMD = SDM_model.model_load("../model/SDM_0.1_9000_.txt")
    print('训练完成：')
    for img in train_imgs[-10:]:
        test_SMD.predict_test(img)