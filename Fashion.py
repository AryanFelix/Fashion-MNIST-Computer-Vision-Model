
# Aryan Felix

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 
60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image,
associated with a label from 10 classes.
'''
x=tf.keras.datasets.fashion_mnist
'''
The inbuilt method 'load_data()' automatically loads 60,000 values into the 'train_set' and 'train_labels'
and 10,000 values into 'test_set' and 'test_labels.'
'''
(train_set,train_label),(test_set,test_label)=x.load_data()
np.set_printoptions(linewidth=200) #Sets width of each line to 200.
plt.imshow(train_set[0]) #Converts image into RGB values ranging from 0-255.
print('\n\n')
print(train_set[0])
print('\n\n')
'''
We use integer labels to differentiate between clothes mainly for two reasons :-
1. It is easier for the machine to process integers rather that strings.
2. To  bypass the international language barrier.
'''
print("Assigned label to above shown article of clothing = {}".format(train_label[0]))
#Let us normalize the training and the test sets for easy and faster processing.
train_set=train_set/255
test_set=test_set/255
#Time to create the model and its neural network
'''
Flatten() : Converts the Multi-Dimension Array to 1D
Dense() : Density of neurons in the particular layer
relu : If x>0, returns x else returns 0
softmax : Assigns 1 to the largest element of the array and 0 to the rest
'''
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
#Training the model
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_set,train_label,epochs=5)
#Testing the model
model.evaluate(test_set,test_label)