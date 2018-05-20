#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 14:09:50 2018

@author: Francisco Antunez
"""

import cv2
import keras
import numpy as np
#from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import threading


model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

cap=cv2.VideoCapture(0)
Z = np.random.random((480, 480, 3)) * 10
while (True):
    thing,vid = cap.read()
    vid = vid[:480, :480]

    vid = vid
    vid = cv2.resize(vid, (224, 224), interpolation = cv2.INTER_LINEAR)
    cv2.imshow('my image', vid)
    vid = vid.reshape(-1,vid.shape[0],vid.shape[1],vid.shape[2])
    pred = model.predict(vid)
    threading.Thread(print(decode_predictions(pred)[0])).start()
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break
        cv2.destroyAllWindows()
        cap.release()

print(Z)
#224x224
#model.predict(img)
