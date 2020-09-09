import abc
import numpy as np
import logging
from GDBT_hog.decision_tree_hog import Decision_tree,getdata
from data_preprocess.pic_preprocess import  showpic
import copy
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
import pickle


class AbstractBaseGradientBoosting(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def predict(self, data):
        pass


class BaseGradientBoosting(AbstractBaseGradientBoosting):

    def __init__(self, learning_rate, n_trees, max_depth,
                 lr_decay=None):  # lr_decay学习率衰减   每过lr_decay个循环  学习率减少0.1
        super().__init__()
        self.learning_rate = learning_rate
        self.learning_rate_cp = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.features = None
        self.trees = {}
        self.f_value = {}
        self.lr_decay = lr_decay

    def fit(self, all_features, target, test_features=None, test_label=None):
        """
        :param data: pandas.DataFrame, the features data of train training
        """
        # 初始化 f_0(x)
        # 对于平方损失来说，初始化 f_0(x) 就是 y 的均值

        self.f_value[0] = target.mean(axis=0)
        for i in range(len(all_features) - 1):
            self.f_value[0] = np.vstack((self.f_value[0], target.mean(axis=0)))
        # 对 m = 1, 2, ..., M

        train_loss_list = []
        test_loss_list = []
        for iter in range(1, self.n_trees + 1):
            if self.lr_decay != None:
                if iter % self.lr_decay == 0:
                    if self.learning_rate >= 0.2:
                        self.learning_rate -= 0.1
                    else:
                        self.learning_rate *= 0.1

            # 计算负梯度--对于平方误差来说就是残差

            res_target = target - self.f_value[iter - 1]

            self.trees[iter] = Decision_tree(all_data=all_features, target=res_target, max_depth=self.max_depth,
                                             logger=logger).nodelist
            self.f_value[iter] = copy.deepcopy(self.f_value[iter - 1])
            for idx,feature in enumerate(all_features):
                self.f_value[iter][idx] = self.f_value[iter][idx] + self.learning_rate * (
                    self.trees[iter][0].predict(feature))

            train_loss = 0
            for idx, feature in enumerate(all_features):
                pre_label = self.predict(feature)
                train_loss += np.mean(np.square(target[idx] - pre_label))
            train_loss = train_loss / len(all_features)
            train_loss_list.append(train_loss)
            print("iter : {}  train_loss : {}".format(iter, train_loss))
            if test_imgs != None:  # 如果加入测试数据，利用并行计算产生预测误差 不用并行了  发现并行速度更慢

                test_loss = 0
                for idx, feature in enumerate(test_features):
                    pre_label = self.predict(feature)
                    test_loss += np.mean(np.square(test_label[idx] - pre_label))
                test_loss = test_loss / len(test_imgs)
                print("test_loss", test_loss)
                test_loss_list.append(test_loss)
            x_axis = range(len(train_loss_list))
            plt.plot(x_axis, train_loss_list, label='train_loss')  # Plot some data on the (implicit) axes.
            plt.plot(x_axis, test_loss_list, label='test_loss')  # etc.
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.title("Loss plot")
            plt.legend()
            plt.savefig('../log/GDBThog_{}_{}_{}_loss.jpg'.format(self.learning_rate_cp, self.n_trees, self.max_depth))
            plt.close('all')
            model_path = '../model/GDBThog_{}_{}_{}.txt'.format(self.learning_rate_cp, self.n_trees, self.max_depth)
            self.model_save(model_path)

    def predict(self, feature):

        result = copy.deepcopy(self.f_value[0][0])
        points1 = np.array(result).reshape(-1, 2)
        points1 = [(int(point[0]), int(point[1])) for point in points1]
        # print(points1)
        # showpic(img,points1)
        lr = self.learning_rate_cp
        for iter in range(1, len(self.trees) + 1):
            if self.lr_decay != None:
                if iter % self.lr_decay == 0:
                    if lr >= 0.2:
                        lr -= 0.1
                    else:
                        lr *= 0.1

            result += lr * self.trees[iter][0].predict(feature)
        # result +=  np.array([self.learning_rate * tree[0].predict(img) for tree in self.trees.values()]).sum(axis=0)

        points1 = np.array(result).reshape(-1, 2)
        points1 = [(int(point[0]), int(point[1])) for point in points1]
        # print(result)
        # showpic(img, points1)
        return result

    def model_save(self, path):

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def model_load(cls, path):
        with open(path, 'rb')as f:
            GDBT = pickle.load(f)
        return GDBT


if __name__ == "__main__":
    train_path = "D:\\wangqiang\\source\\pickeddataset\\trainset"
    test_path = "D:\\wangqiang\\source\\pickeddataset\\testset"

    LEARNING_RATE = 0.5
    TREE_NUMBER = 600

    MAX_DEPTH = 3
    lr_decay = 100
    train_imgs, train_landmarks,train_features = getdata(train_path, 2000)
    print(np.array(train_landmarks).shape)
    train_label = np.array(train_landmarks).reshape(-1, 24)  # (-1, 136)
    test_imgs, test_landmarks,test_features = getdata(test_path, 200)
    test_label = np.array(test_landmarks).reshape(-1, 24)

    stime = time.time()
    testB = BaseGradientBoosting(LEARNING_RATE, TREE_NUMBER, MAX_DEPTH, lr_decay)
    testB.fit(train_features, train_label, test_features, test_label)
    etime = time.time()
    print('costtime: ', etime - stime)

    # for idx ,img in enumerate(train_imgs):
    #     testB.predict(img)

    # testB = BaseGradientBoosting.model_load(model_path)



