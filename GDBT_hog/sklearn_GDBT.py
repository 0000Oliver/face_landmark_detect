
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection ,metrics
from sklearn.model_selection import GridSearchCV,cross_validate,cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import pandas as pd
import pickle
from GDBT_hog.decision_tree_hog import Decision_tree,getdata

def modelfit(alg,X,y,):
    #alg.fit(X,y)


    pred = alg.predict(X)
    total_err = 0
    for i in range(pred.shape[0]):
        print(pred[i], y[i])
        err = (pred[i] - y[i]) #/ y[i]
        total_err += err * err
    print(total_err / pred.shape[0])


if __name__ =="__main__":
    train_path = "D:\\wangqiang\\source\\pickeddataset\\trainset"
    test_path = "D:\\wangqiang\\source\\pickeddataset\\testset"

    LEARNING_RATE = 0.5
    TREE_NUMBER = 600

    MAX_DEPTH = 3
    lr_decay = 100
    train_imgs, train_landmarks, train_features = getdata(train_path, 2000)
    print(np.array(train_landmarks).shape)
    train_label = np.array(train_landmarks).reshape(-1, 24)  # (-1, 136)
    test_imgs, test_landmarks, test_features = getdata(test_path, 200)
    test_label = np.array(test_landmarks).reshape(-1, 24)

    gbms = []
    for idx in range(24):
        print("第{}棵树".format(idx))
        gbm = GradientBoostingRegressor( loss='ls',learning_rate=0.1, n_estimators=200, subsample=1, max_depth=3,
                                          min_samples_split=40, min_samples_leaf=1, max_features=None, random_state=None)
        y_train = train_label[:,idx]
        gbm.fit(train_features,y_train)
        y_test = test_label[:,idx]
        modelfit(gbm, test_features, y_test)
        gbms.append(gbm)



