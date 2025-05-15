#Assignment 3: Load and Evaluate Trained Model
#Weerawan Pasomthong 644357

import keras
import numpy as np
import tensorflow as tf
from seaborn import heatmap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report




#Load data
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


mean = np.mean(train_x,axis=(0,1,2,3))
std = np.std(train_x,axis=(0,1,2,3))
test_x = (test_x-mean)/(std+1e-7)



# load json file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')

#Model Description
print("Number of Layers is:",len(loaded_model.layers))
loaded_model.summary()


#Confusion Matrix
pred = loaded_model.predict(x=test_x, verbose=0)
predictions_for_cm = pred.argmax(1)
cm = confusion_matrix(test_y, predictions_for_cm)
plt.figure(figsize=(8, 8))
pred = loaded_model.predict(x=test_x, verbose=0)
heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes)
plt.show()

#Classification Report
y_pred = loaded_model.predict(test_x, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(test_y, y_pred_bool))



def Visualize_Weights(model,number_of_layer):
    #Display Descriptive Values
    print("{} {}{}".format("Weights of Layer",number_of_layer,':'))
    values = model.layers[number_of_layer].get_weights()[0]
    print('Mean: ',values.mean())
    print('Max: ', values.max())
    print('Min: ', values.min())
    print('Standard Dev: ', values.std())
    print('-------------------------------------------------------')


    #get weights
    filters, biases = model.layers[number_of_layer].get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 7, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 3, ix)
            plt.suptitle("{} {}".format("First 7 Filters of Layer",number_of_layer))
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    return plt.show()


#visualize the first and last convolutional layer
Visualize_Weights(loaded_model,0)
Visualize_Weights(loaded_model,11)









