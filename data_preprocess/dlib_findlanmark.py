import sys
import  dlib
from skimage import io
import os
import numpy as np
import cv2
train_root = 'D:\\wangqiang\\source\\ibug_300W_large_face_landmark_dataset.tar\\ibug_300W_large_face_landmark_dataset\\helen\\trainset'
dirlist = os.listdir(train_root)
dirlist = [dirname.split(".")[0] for dirname in dirlist]
codelist = [dirname[:-7] if "mirror" in dirname else dirname  for dirname in dirlist]
codelist = list(set(codelist))
predictor_path='.\\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
window = dlib.image_window()
img = io.imread(os.path.join(train_root, codelist[0] + '.jpg'))

dets = detector(img,1)
print("Numer of faces detected:{}".format(len(dets)))
for i , d in enumerate(dets):
    print("Detection{}: left:{} Top:{} Right:{} Bottom:{}".format(i,d.left(),d.top(),d.right(),d.bottom))
    shape = predictor(img,d)

    landmark = [[p.x,p.y] for p in shape.parts()]
    # print(landmark)
    print('face_landmark:')
    window.add_overlay(shape)
    for idx, point in enumerate(landmark):
        # print(point[0][0][0])
        print(point[0])
        pos = (point[0], point[1])
        cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.3, color=(0, 255, 0))


# window.clear_overlay()
window.set_image(img)
window.add_overlay(dets)


dlib.hit_enter_to_continue()