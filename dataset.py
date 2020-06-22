#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:05:29 2019

@author: kartik
"""


import glob
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import shutil
import math
from sklearn.svm import SVC
from skimage.feature import hog

def get_dataset(path):
    data = []
    for folder in glob.glob(path+"\\*"):
        if(folder.split('\\')[-1] == 'dataset.py'):
            continue
        f = []
        for image in glob.glob(folder+"\\*"):
            img = cv2.imread(image,0)
            f.append(img)
        data.append(f)
    return data

def crop_images(images, folder_names):
    flag = 0
    #folder_names = ['AngryCrop', 'FearCrop', 'HappyCrop', 'SadCrop', 'SurpriseCrop', 'TestCrop']
    for i in range(6):
        if flag==0:
            flag = 1
            #print(os.path.isdir(path+'\\'+folder_name[i]))
            if os.path.isdir(path+'\\'+folder_names[i]):
                shutil.rmtree(path+"\\"+folder_names[i])
            os.mkdir(path+'\\'+folder_names[i])
            
        for j in range(len(images[i])):
            gray = images[i][j]
            #reference: https://www.digitalocean.com/community/tutorials/how-to-detect-and-extract-faces-from-an-image-with-opencv-and-python
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3,  minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_color = gray[y:y + h, x:x + w]
                #print("[INFO] Object found. Saving locally.")
                os.chdir(path+'\\'+folder_names[i])
                cv2.imwrite(os.getcwd()+"\\"+folder_names[i] + str(w) + str(h) + folder_names[i] +'.jpg', roi_color)
                os.chdir("..")
        flag = 0
        
def split_dataset(path, folder_names):
    x_train, x_test = [], []
    y_train = []
    test = 'TestCrop'
    folder_names.remove(folder_names[-1])
    #names = [i.split('\\')[-1] for i in glob.glob(path+"\\*")]
    for folder in glob.glob(path+"\\*"):
        name = folder.split('\\')[-1]
        #print(name)
        if(name == 'dataset.py'):
            continue
        elif name in folder_names:
            f = []
            for image in glob.glob(path+'\\'+name+"\\*"):
                img = cv2.imread(image,0)
                f.append(img)
                y_train.append(name.split('C')[0])
            #x_train.append(f)       #change to x_train+=f if you want all images in continuation.
            x_train+=f
        if name == test:
            f=[]
            for image in glob.glob(path+'\\'+name+"\\*"):
                img = cv2.imread(image,0)
                f.append(img)
            x_test+=f
    return x_train, x_test, y_train

def check_folder_exist(path, folder_names):
    i = 0
    for folder in glob.glob(path+'\\*'):
        if os.path.isdir(folder) and folder.split('\\')[-1] in folder_names:
            print("Deleting...")
            shutil.rmtree(folder)
        i+=1
    
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def apply_filters(x_train):
    median, m_psnr, gaussian, g_psnr = [], [], [], []
    for img in x_train:
        g_img = cv2.GaussianBlur(img,(5,5),0)
        gaussian.append(g_img)
        g_psnr.append(psnr(img, g_img))
        m_img = cv2.medianBlur(img,5)
        median.append(m_img)
        m_psnr.append(psnr(img, m_img))
    return median, gaussian, m_psnr, g_psnr

def apply_hog(x_train, y_train, folder_names):
    x_trainf, y_trainf, hog_img  = [], [], []
    for img in x_train:
        x_trainf.append(img.flatten())
        hog_features = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm="L2")
        hog_img.append(hog_features)
        
    return np.array(x_trainf), np.array(hog_img)
        
def svm(x, y):
    model = SVC(kernel = 'linear')
    model.fit(x, y)
    return model

folder_names = ['AngryCrop', 'FearCrop', 'HappyCrop', 'SadCrop', 'SurpriseCrop', 'TestCrop']
path = os.getcwd()
check_folder_exist(path, folder_names)
images = get_dataset(path)
cropped = crop_images(images, folder_names)
x_train, x_test, y_train = split_dataset(path, folder_names)
median, gaussian, m_psnr, g_psnr = apply_filters(x_train)

x_trainf, hog_train = apply_hog(x_train, y_train, folder_names)
hog_train_T = hog_train.T
model = svm(hog_train, y_train)
#x_testf, hog_test = apply_hog(x_test, y_test, folder_names)
#print(model.score(hog_test, y_test))