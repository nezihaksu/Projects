import numpy as np

#To silence tf info.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds

dataset,info = tfds.load(name = 'mnist',with_info = True,as_supervised = True)
train,test = dataset['train'],dataset['test']

num_train = info.splits['train'].num_examples
num_test = info.splits['test'].num_examples


def scaler(image,label):
	image =  tf.cast(image,tf.float32)
	image /= 255
	return image,label

num_val_samples = 0.1*num_train
num_val_samples = tf.cast(num_val_samples,tf.int64)

num_test = tf.cast(num_test,tf.int64)

scaled_train_validation = train.map(scaler)
scaled_test = test.map(scaler)

buffer_size = 10_000
batch_size = 150

scaled_train_validation = scaled_train_validation.shuffle(buffer_size)
validation_data = scaled_train_validation.take(num_val_samples)
train_data = scaled_train_validation.skip(num_val_samples)

train_data = train_data.batch(batch_size)
validation_data = validation_data.batch(batch_size)

test_data = scaled_test.batch(batch_size)

validation_input,validation_target = next(iter(validation_data))


input_shape = 784
learning_rate = 0.001
num_epochs = 15
#batch_size = 100
n_neuron = 100
image_tensor_shape = (28,28,1)
n_outputs = 10


model_a = Sequential()
model_a.add(Flatten(input_shape = image_tensor_shape))
model_a.add(Dense(n_neuron,activation = 'relu',kernel_initializer = 'he_normal'))
model_a.add(BatchNormalization())
model_a.add(Dense(n_neuron,activation = 'relu',kernel_initializer = 'he_normal'))
model_a.add(BatchNormalization())
model_a.add(Dense(n_outputs,activation = 'softmax'))

model_a.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)

model_a.fit(train_data,epochs = num_epochs,validation_data = (validation_input,validation_target),verbose = 2)

test_loss,test_accuracy = model_a.evaluate(test_data)

print("Test Accuracy: {},Test Loss{}".format(test_accuracy,test_loss))

tf.saved_model.save(model_a,r"C:\Users\nezih\Desktop")


#Model accuracy is 99.68,test accuracy is 97.60.

