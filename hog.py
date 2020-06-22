
# coding: utf-8

# In[115]:


import glob
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import shutil
from skimage.feature import hog


# In[2]:


def get_dataset(path):
    data = []
    for folder in glob.glob(path+"\\*"):
        if(folder.split('\\')[-1] == 'dataset.py'):
            continue
        f = []
        for image in glob.glob(folder+"\\*"):
            img = cv2.imread(image)
            f.append(img)
        data.append(f)
    return data


# In[3]:


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


# In[4]:


path = os.getcwd()
images = get_dataset(path)
cropped = crop_images(images)


# In[ ]:


#get_ipython().system('dir')


# In[43]:


FaceFolders = ['AngryCrop','FearCrop', 'HappyCrop', 'SadCrop', 'SurpriseCrop']
FaceLabels = 'X'
#FaceHOGs = np.zeros[1,7128]


# In[91]:


trainimages =[]
trainlables = []
hog_img =[]
for i in range (len(FaceFolders)):
    Emotion = FaceFolders[i]
    #print(Emotion)
    TrainingImages = Emotion
    #Training_Folder = "F:\\IIITD\\DIP\\DIP Project\\code_pao\\EE368 Final Project MATLAB JPao\\" + str(TrainingImages)
    #Training_Folder = "F:\\IIITD\\DIP\\DIP Project\\code_pao\\EE368 Final Project MATLAB JPao\\TotalCrop\\" + str(TrainingImages)
    Training_Folder = []
    TrainingImages = os.getcwd()
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


# In[92]:


hog_t = hog_img.T


# In[93]:


print(hog_t.shape)


# In[94]:


print (trainlables)


# In[105]:


print(hog_t.shape)
print(trainlables.shape)
from sklearn import svm
svc = svm.SVC(kernel= 'linear')
svc.fit(hog_img, trainlables)


# In[106]:


from sklearn import metrics
score_train_ovr = svc.score(hog_img, trainlables)
print("Train Accuracy of SVC OVR:",score_train_ovr)


# In[107]:


testimages =[]
testlables = []
hogtest_img =[]
#test_dir = "F:\\IIITD\\DIP\\DIP Project\\code_pao\\EE368 Final Project MATLAB JPao\\TestCrop\\TestFacesCrop" 
test_dir = os.getcwd()
for i in range (1,33):
    
    test_img = test_dir + str(i) + str('.png') # Enter Directory of all images 
    print (test_img)
    test_image = cv2.imread(test_img,0)
    print(type(test_image))
    plt.imshow(test_image, cmap=plt.cm.gray)
    if test_image is not None:
        
        print(test_image.shape)       
        test_image.flatten()
        #testimages.append(test_image)
        thogfeatures = hog(test_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm="L2")
        #thogfeatures = hog(test_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2")
        hogtest_img.append(thogfeatures)
        
testimages = np.array(testimages)
#trainlables =np.asarray(trainlables)
hogtest_img = np.array(hogtest_img)
print(testimages.shape)
#print (trainlables.shape)
print(hogtest_img.shape)
#plt.imshow(hogtest_img[0])


# In[108]:


testlables = [0,3,3,4,3,1,2,3,4,3,2,2,1,4,0,2,2,4,0,0,2,2,0,0,1,1,2,2,3,3,4,4]


# In[109]:


print(testlables)


# In[110]:


testlables =np.array(testlables)


# In[111]:


print(testlables)


# In[112]:


y_pred = svc.predict(hogtest_img)
svc_score = metrics.accuracy_score(testlables, y_pred)
print(svc_score)


# In[113]:


y_pred_train = svc.predict(hog_img)
print(y_pred_train)


# In[114]:


print(y_pred)

