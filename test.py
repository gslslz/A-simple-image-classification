# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 08:55:38 2018

@author: Conan
"""

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

norm_size = 256
class_all=('glacier','rock','urban','water','wetland','wood')
   
def predict(image_path):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model("vgg16model.h5")
    
    #load the image
    image = cv2.imread(image_path)
    orig = image.copy()
     
    # pre-process the image for classification
    image = cv2.resize(image, (norm_size, norm_size))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
     
    # classify the input image
    result = model.predict(image)[0]
    #print (result.shape)
    proba = np.max(result)
    for s in range(0,6):
    	if np.where(result==proba)[0]==s:
             label = class_all[s]
             break
    label = "{}: {:.2f}%".format(label, proba * 100)
    print(label)
    
    if 'show': 
        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25),cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)       
        # show the output image
        cv2.imshow("Output", output)
        cv2.waitKey(0)


#python predict.py --model traffic_sign.model -i ../2.png -s
if __name__ == '__main__':
    predict('test01.jpg')
    
    
