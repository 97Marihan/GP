import numpy as np
import pickle
import cv2
import os
from glob import glob
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

num_classes = 5  # the five classes that will be classified
data = []   # l images
labels = []

data_dir = "Dataset\\face-shape-dataset\\FaceShape_Dataset\\training_set"
data_list = glob(data_dir+'\\*')    # for loading the names to be used in one hot encoding as output layer


for folder in data_list: # l folder l gowah l 5 folders
    for file in glob(folder+'\\*'): # l files l soghyra l gowa l folder l keber (oval,round,squared,diamond,heart)
        try:
            img = cv2.imread(file) # put the files into variable named img
            image = cv2.resize(img, (300, 300)) #resize l images kolaha w han7otaha gowa variable esmo image
            data.append(image.reshape(300, 300, 3)) #reshape  the 3 RGB layers
            labels.append(folder.split('\\')[-1]) # ??
        except:
            print(file)

            #### one hot- encoding hat5le l class l image taba3o b 1 w ba2y l classes b 0
dataX = np.array(data, dtype="float") / 255.0      # Normalize all pixels
labels = np.array(labels)

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dataY = np_utils.to_categorical(encoded_Y)   # one-hot encoding

with open('data_training.pkl', 'wb') as file_id:
    pickle.dump([dataX, dataY], file_id)
file_id.close()
