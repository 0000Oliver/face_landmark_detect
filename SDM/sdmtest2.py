from data_preprocess.pic_preprocess import getdata,showpic,getdata2
from SDM.Hog_feature import get_hog_feature,landmark_to_label,label_to_landmark
import numpy as np
from SDM.SMD import SDM_model
if __name__ =="__main__":
    LEARNING_RATE = 0.3
    TOTAL_NUMBER = 1000
    TRAIN_NUMBER = int(TOTAL_NUMBER * 0.8)
    TEST_NUMBER = int(TOTAL_NUMBER * 0.2)
    train_imgs, train_landmarks = getdata2(TOTAL_NUMBER)
    train_labels = landmark_to_label(train_landmarks)
    train_labels = np.array(train_labels)
    print(len(train_imgs))


    # test_SMD = SDM_model(LEARNING_RATE,pca_number=None,lr_decay=1)
    # test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    # test_SMD = SDM_model(LEARNING_RATE,pca_number=None,lr_decay=2)
    # test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    test_SMD = SDM_model(LEARNING_RATE,pca_number=None,lr_decay=3)
    test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    test_SMD = SDM_model(LEARNING_RATE,pca_number=None,lr_decay=4)
    test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    test_SMD = SDM_model(LEARNING_RATE,pca_number=None,lr_decay=5)
    test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    # test_SMD = SDM_model(LEARNING_RATE,pca_number=600)
    # test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])
    # test_SMD = SDM_model(LEARNING_RATE,pca_number=700)
    # test_SMD.fit(train_imgs[:TRAIN_NUMBER],train_labels[:TRAIN_NUMBER],train_imgs[-TEST_NUMBER:],train_labels[-TEST_NUMBER:])