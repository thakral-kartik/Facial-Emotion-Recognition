import glob
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import shutil
from skimage.feature import hog

def get_dataset(path):
    data = []
    for folder in glob.glob(path+"\\*"):
        if(folder.split('\\')[-1] in ['dataset.py', 'Final+DIP+Project.py', 'main.py']):
            continue
        f = []
        for image in glob.glob(folder+"\\*"):
            img = cv2.imread(image)
            f.append(img)
        data.append(f)
    return data
	
def crop_images(images):
    flag = 0
    folder_name = ['AngryCrop', 'FearCrop', 'HappyCrop', 'SadCrop', 'SurpriseCrop', 'TestCrop']
    #path1 = 'C:\\opencv\\build\\etc\\haarcascades'
    path1 = os.getcwd()
    for i in range(6):
        if flag==0:
            flag = 1
            #print(os.path.isdir(path+'\\'+folder_name[i]))
            if os.path.isdir(path+'\\'+folder_name[i]):
                shutil.rmtree(path+"\\"+folder_name[i])
            os.mkdir(path+'\\'+folder_name[i])
            
        for j in range(len(images[i])):
            gray = images[i][j]
            faceCascade = cv2.CascadeClassifier(os.path.join(path1, "haarcascade_frontalface_default.xml"))
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3,  minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_color = gray[y:y + h, x:x + w]
                #print("[INFO] Object found. Saving locally.")
                os.chdir(path+'\\'+folder_name[i])
                cv2.imwrite(os.getcwd()+"\\"+folder_name[i] + str(w) + str(h) + folder_name[i] +'.jpg', roi_color)
                os.chdir("..")
        flag = 0

def apply_hog():
    FaceFolders = ['AngryCrop','FearCrop', 'HappyCrop', 'SadCrop', 'SurpriseCrop']
    FaceLabels = 'X'
    
    trainimages =[]
    trainlables = []
    hog_img =[]
    for i in range (len(FaceFolders)):
        Emotion = FaceFolders[i]
        #print(Emotion)
        TrainingImages = Emotion
        #Training_Folder = "F:\\IIITD\\DIP\\DIP Project\\code_pao\\EE368 Final Project MATLAB JPao\\" + str(TrainingImages)
        Training_Folder = "F:\\IIITD\\DIP\\DIP Project\\code_pao\\EE368 Final Project MATLAB JPao\\TotalCrop\\" + str(TrainingImages)
        print(Training_Folder)
        for j in range (len(Training_Folder)):
            input_img = Training_Folder + str(j) + str('.png')
            print (input_img)
            img = cv2.imread(input_img,0)
            print(img)
            if img is not None:
                print(img.shape)       
                img.flatten()
                trainimages.append(img)
                trainlables.append(i)
                print("type =",type(img))
                hogfeatures,h = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm="L2")
                hog_img.append(hogfeatures)
    trainimages = np.array(trainimages)
    trainlables =np.array(trainlables)
    hog_img = np.array(hog_img)
    print(trainimages.shape)
    print (trainlables.shape)
    print(hog_img.shape)        




path = os.getcwd()
images = get_dataset(path)
cropped = crop_images(images)
#train_images, train_labels = apply_hog()