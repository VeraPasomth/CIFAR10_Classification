#Assignment 3 Multiclassification CNN
#Weerawan Pasomthong 644357


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import adam_v2
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten,BatchNormalization





#Load Data
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


#Data Normalization
mean = np.mean(train_x,axis=(0,1,2,3))
std = np.std(train_x,axis=(0,1,2,3))
train_x = (train_x-mean)/(std+1e-7)
test_x = (test_x-mean)/(std+1e-7)


input_shape = (32,32,3)

test_y = test_y.reshape(-1)
train_y = train_y.reshape(-1)

label_as_binary = LabelBinarizer()
train_y = label_as_binary.fit_transform(train_y)




def CovnetModel():
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10,activation='softmax'))

    return model



if __name__ == "__main__":
    #Create Model
    model = CovnetModel()
    model.compile(loss='categorical_crossentropy',optimizer=adam_v2.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    #Data Augmentation
    datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,)
    datagen.fit(train_x)

    #Train Model
    history = model.fit(train_x,train_y, batch_size=128, epochs=30,verbose=1,
                        validation_split=0.1, validation_data=(test_x,test_y))
    model.summary()


    #Display Results
    plt.subplot(2, 1, 2)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(history.history['accuracy'],'b')
    plt.plot(history.history['val_accuracy'],'r')
    plt.title('Learning Curve: Training and Test Accuracy')
    plt.legend(['Training', 'Test'], loc='lower right')
    plt.show()

    #Save Model
    model_json = model.to_json()
    with open('New_model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights('New_model.h5')


