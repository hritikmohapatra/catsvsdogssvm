# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 21:44:59 2021

@author: HRITIK MOHAPATRA
"""

#Importing Required Libraries

import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import pickle 
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#Assigning The Data directory to a variable
dir = r"D:\Hritik\Study\Machine learning\Project\Dogs vs Cats\dataset"
categories = ['cats', 'dogs'] 

#Store the pictures in List and Resize it

data = []

for category in categories: 
    path = os.path.join(dir, category)
    label = categories.index(category) 
    
    for img in os.listdir(path): 
        imgpath = os.path.join(path,img) 
        pet_img = cv2.imread(imgpath, 0) 
        pet_img = cv2.resize(pet_img, (50,50)) 
        image = np.array(pet_img).flatten() 
        data.append([image, label]) 


print(len(data))

#Create the Pickle using Data
pick_in = open('data1.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)

#Separating data into Features and labels
features = []
labels = []

for feature , label in data:
    features.append(feature)
    labels.append(label)

#Dividing data into Training and Testing data
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.25)


#Assign Model and train the data
model = SVC(C=1, kernel= 'poly', gamma= 'auto')
model.fit(xtrain, ytrain)

#Save the model
pick = open('model.sav', 'wb')
pickle.dump(model, pick)
pick.close()

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

categories = ['cats', 'dogs'] 

print('Accuracy: ', accuracy)
print('Prediction is : ',categories[prediction[0]])

mypet = xtest[0].reshape(50,50)
plt.imshow(mypet, cmap='gray')
plt.show






