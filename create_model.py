from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import keras
import tensorflow as tf
from time import time

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('C:\\Users\\AJ\\Desktop\\img recong\\devanagari-character-recognition\\train',target_size = (32,32),batch_size = 100)
test_set = test_datagen.flow_from_directory('C:\\Users\\AJ\\Desktop\\img recong\\devanagari-character-recognition\\test',target_size = (32,32),batch_size = 100)

num_classes = 46

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(32,32,3), activation='relu'))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(units=64, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(training_set,steps_per_epoch = 100,
epochs = 50,
validation_data = test_set,validation_steps = 100,
callbacks=[keras.callbacks.TensorBoard(log_dir="C:\\Users\\AJ\\Desktop\\img recong\\devanagari-character-recognition\\logs".format(time()),
write_graph=True, write_images=True)])
#sess = tf.Session()
sess = tf.compat.v1.Session()

file_writer = tf.summary.create_file_writer('C:\\Users\\AJ\\Desktop\\img recong\\devanagari-character-recognition\\logs')
model.save('devnagri_character_model.h5')