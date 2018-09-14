from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import cv2

PATH_MODEL = "model/number_classification_model.hd5"

# load the model
model = load_model(PATH_MODEL)

# set camera number
cap = cv2.VideoCapture(1) # 480 x 640

while(True):
    # capture frame
    ret, frame = cap.read()

    # crop the frame x,y,w,h = 170,0,300,480
    frame = frame[0:0+480, 170:170+300]
    
    # resize the frame to meet the input format requirement of the model
    image = cv2.resize(frame,(25,25))
    image = np.array(image)

    # convert the image format
    image = image.transpose(2, 0, 1)
    image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
    
    # get the result from the model
    result = model.predict_classes(np.array([image / 255.]))
    
    # put text on the frame
    cv2.putText(frame,"This is "+str(result[0])+".",(10,30),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2)
    
    # display the frame
    cv2.imshow('frame', frame)
    
    print("result: ", result[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything is done, release the capture
cap.release()
cv2.destroyAllWindows()




