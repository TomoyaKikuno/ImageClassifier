from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

DIR_TRAIN = "data/numbers/train"
DIR_TEST  = "data/numbers/test"
PATH_MODEL = "model/number_classification_model.hd5"

# create dataset for machine learning
image_list = []
label_list = []

# load the files under ./data/train/numbers/
for file in os.listdir(DIR_TRAIN):
    if file != ".DS_Store": # I'm using mac os... ignore this when you run this code on other OS

        # get the label prefix of the file name and add this to label_list
        label = file[0:file.find('_')]
        label_list.append(label)
        
        # get the file path
        filepath = DIR_TRAIN + "/" + file

        # convert the image format into 25x25 pixel format, and load the image as a 2D array of 25x25 sets with [R,G,B] components
        # each R,G,B has a value from 0 to 255
        image = np.array(Image.open(filepath).resize((25,25)))
        print(filepath)

        # convert the array format into the array like [[Red array], [Green array], [Blue array]]
        image = image.transpose(2, 0, 1)

        # convert the array format again into the flat 1D array format where red values, green values, and blue values line up in a row
        image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]

        # append the array to image_list (each value ranges from 0 to 1)
        image_list.append(image / 255.)

# convert image_list into numpy array to handle this with keras
image_list = np.array(image_list)


# convert the label array that consists of values from 0 to 1
# cf) 0 -> [1,0], 1 -> [0,1]
Y = to_categorical(label_list)

# compose neural networks by configuring a model
# in the following codes specific layers are added to the model
# Accuracy of this model can be optimized by changing the following codes
model = Sequential()
model.add(Dense(200, input_dim=1875)) # input dim = 3(r,g,b) * 25 * 25
model.add(Activation("relu"))
#model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
#model.add(Dropout(0.2))

model.add(Dense(10)) # number of patterns
model.add(Activation("softmax"))

# use Adam for the optimizer
opt = Adam(lr=0.001)

# compile the model
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# execute the learning process
model.fit(image_list, Y, epochs=200, batch_size=100, validation_split=0)

# save the model
model.save(PATH_MODEL)

# calculate and show the accuracy using test images
count_total = 0.
count_correct = 0.

print("")
print("-----test-------")
print("")

for file in os.listdir(DIR_TEST):
    if file != ".DS_Store": # I'm using mac os... ignore this when you run this code on other OS

        # get the label prefix of the file name
        label = file[0:file.find('_')]
        label_list.append(label)
    
        # get the file path
        filepath = DIR_TEST + "/" + file

        # load the test image
        image = np.array(Image.open(filepath).resize((25,25)))
        print(filepath)

        # convert the iamge format
        image = image.transpose(2, 0, 1)
        image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
        result = model.predict_classes(np.array([image / 255.]))
        print("label:", label, "result:", result[0])

        count_total += 1.
        
        if str(label) == str(result[0]):
            count_correct += 1.

print("count_total: ", count_total)
print("count_correct: ", count_correct)
print("accuracy rate: ", count_correct / count_total * 100, "%")













