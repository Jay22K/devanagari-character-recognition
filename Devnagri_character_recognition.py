from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

training_set = train_datagen.flow_from_directory('C:\\Users\\AJ\\Desktop\\img recong\\devanagari-character-recognition\\train',
target_size = (32, 32),
batch_size = 32)

classes = training_set.class_indices
#print(classes)
classifier = load_model('C:\\Users\\AJ\\Desktop\\img recong\\devnagri_character_model.h5')
classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

testImage = keras.utils.load_img(input("image name & path:- ")+'.png')
a = np.resize(testImage, (32,32,3))
testimage = keras.utils.img_to_array(a) 
print(testimage)
testimage = np.expand_dims(testimage, axis = 0)
print(testimage)


plt.title("Input Image")
plt.imshow(a, cmap=plt.cm.binary)
plt.show()

#prediction = classifier.predict_classes(testimage)
prediction = np.argmax(classifier.predict(testimage), axis = 1)

for key,value in classes.items():
    if value == prediction:
       print(prediction)
       print("The Image is : ",key)

