from data_preprocess.pic_preprocess import getdata,showpic
from SDM.Hog_feature import get_hog_feature,landmark_to_label,label_to_landmark
import  numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import time
import logging



class SDM_model():#结构方面和SDM 原方法完全一样，在计算下降方向的部分尝试新的方法。
    def __init__(self,leaning_rate):
        self.learning_rate = leaning_rate
        self.f_value={}
        self.derection={}
        self.loss = 100
        self.avg_phi_star = None
        logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='test.log',
                filemode='w')
        self.logger =logging.getLogger()
        # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')#('%(name)-12s: %(levelname)-8s )
        console.setFormatter(formatter)
        #logging.getLogger('').addHandler(console)
        self.logger.addHandler(console)


    def get_phi(self,imgs,label):
        phi_star = [get_hog_feature(imgs[idx], label_to_landmark(label[idx])) for idx in
                    range(len(imgs))] # (4-1)*(4-1)*9*4*12 =3888
        phi_star = np.array(phi_star)
        return phi_star

    def Descent(self,imgs, state, label,phi_star):
        '''
        用最小二乘法求出y=ax+b 中的ab
        :param imgs:
        :param state:
        :param label:
        :return:
        '''
        delta_X = label - state
        phi =self.get_phi(imgs,state)
        A = phi-phi_star
        A = np.mat(A)
        X = np.dot(np.dot(np.linalg.pinv(np.dot(A.T, A)), A.T), delta_X)
        return X, np.dot(A, X)
    def fit(self,imgs,labels,test_imgs = None,test_labels=None):
        self.logger.info("SMD2 开始训练")
        state = []
        for i in range(len(labels)):
            state.append(np.array(labels).mean(axis=0))
        state = np.array(state)
        self.f_value[0] = state
        phi_star = self.get_phi(imgs, labels)
        self.avg_phi_star = phi_star.mean(axis=0)
        train_loss_list = []
        test_loss_list = []
        train_loss = self.compute_loss(state, labels)
        train_loss_list.append(train_loss)
        self.logger.info("train_loss : {}".format(train_loss))
        print("train_loss : ", train_loss)
        if test_imgs != None:

            states = self.predict_batch(test_imgs)
            test_loss = self.compute_loss(states, test_labels)
            test_loss_list.append(test_loss)
            print("test_loss : ", test_loss)

        self.loss = train_loss
        iter = 0

        while self.loss>1:
            print("epoch : ",iter)
            self.derection[iter],strid = self.Descent(imgs,self.f_value[iter],labels,phi_star)
            self.f_value[iter+1] = self.f_value[iter] + self.learning_rate* strid
            train_loss = self.compute_loss(self.f_value[iter+1],labels)
            train_loss_list.append(train_loss)
            self.loss = train_loss
            print("train_loss : ", self.loss)
            if test_imgs != None:
                test_loss = 0
                for idx ,img in enumerate(test_imgs):
                    state = self.predict(img)
                    test_loss+= self.compute_loss(state,test_labels[idx])
                test_loss = test_loss/len(test_imgs)
                test_loss_list.append(test_loss)
                print("test_loss : ", test_loss)


                # states = self.predict_batch(test_imgs)
                # test_loss = self.compute_loss(states,test_labels)
                # test_loss_list.append(test_loss)
                # print("batch_test_loss : ", test_loss)
            iter+=1
            x_axis = range(len(train_loss_list))
            plt.plot(x_axis, train_loss_list, label='train_loss')  # Plot some data on the (implicit) axes.
            if test_imgs != None: plt.plot(x_axis, test_loss_list, label='test_loss')  # etc.
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.title("Loss plot")
            plt.legend()
            figpath = '../log/{}_{}_SDM2_loss.jpg'.format(self.learning_rate,len(imgs))
            if test_imgs != None:figpath = '../log/{}_{}_SDM2_loss_withtest.jpg'.format(self.learning_rate,len(imgs))
            plt.savefig(figpath)
            plt.close('all')
            model_path = '../model/SDM2_{}_{}_.txt'.format(self.learning_rate,len(imgs))

            self.model_save(model_path)

    def compute_strid(self,img, state, decent_map):
        '''
        输入单张图片 和 标注 利用计算出来的每一步方向求出每一步的步伐
        :param imgs:
        :param state:
        :param decent_map:
        :return:
        '''
        R = decent_map
        phi = get_hog_feature(img, label_to_landmark(state))  # (4-1)*(4-1)*9*4*12 =3888
        phi = np.array(phi)
        A = phi-self.avg_phi_star

        return np.dot(A, R)

    def predict(self,img):
        state = [copy.deepcopy(self.f_value[0][0])]
        for i in range(len(self.derection)):
            state+=self.learning_rate*self.compute_strid(img,state,self.derection[i])
        return state
    def predict_test(self,img):
        print("预测")
        showpic(img,label_to_landmark(self.f_value[0][0]))
        state = [copy.deepcopy(self.f_value[0][0])]
        for i in range(len(self.derection)):
            state+=self.learning_rate*self.compute_strid(img,state,self.derection[i])
        showpic(img, label_to_landmark(state))
        return state
    def compute_strid_batch(self,imgs, states, decent_map):
        '''
        输入多张图片 和 标注 利用计算出来的每一步方向求出每一步的步伐
        :param imgs:
        :param state:
        :param decent_map:
        :return:
        '''
        R = decent_map
        phi = [get_hog_feature(imgs[idx], label_to_landmark(states[idx]))for idx in range(len(imgs))]  # (4-1)*(4-1)*9*4*12 =3888
        phi = np.array(phi)
        train_phi_star = [self.avg_phi_star for idx in range(len(imgs))]
        train_phi_star = np.array(train_phi_star)
        A = phi-train_phi_star

        return np.dot(A, R)

    def predict_batch(self,imgs):
        state = [copy.deepcopy(self.f_value[0][0]) for _ in range(len(imgs))]
        for i in range(len(self.derection)):
            state+=self.learning_rate*self.compute_strid_batch(imgs,state,self.derection[i])
        return state

    def compute_loss(self,state, label):
        '''
        计算当前位置和目标位置的误差
        :param state:
        :param label:
        :return:
        '''
        loss = state - label
        return np.square(loss).mean()

    def model_save(self,path):

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def model_load(cls,path):
        with open(path, 'rb')as f:
             SDM = pickle.load(f)
        return  SDM






if __name__ =="__main__":
    SDM_train_path = "D:\\wangqiang\\source\\SMD_dataset\\alltrainset"
    SDM_test_path = "D:\\wangqiang\\source\\SMD_dataset\\alltestset"

    LEARNING_RATE = 0.1

    train_imgs, train_landmarks = getdata(SDM_train_path, 2000)
    train_labels = landmark_to_label(train_landmarks)
    train_labels = np.array(train_labels)
    test_imgs, test_landmarks = getdata(SDM_test_path, 100)
    test_labels = landmark_to_label(test_landmarks)
    test_labels = np.array(test_labels)
    print(train_imgs[0].shape)


    test_SDM = SDM_model(LEARNING_RATE)
    test_SDM.fit(train_imgs,train_labels,test_imgs,test_labels)
    #test_SDM = SDM_model.model_load("../model/SDM_0.1_5400_.txt")
    print('训练完成：')
    for img in test_imgs:
        test_SDM.predict_test(img)