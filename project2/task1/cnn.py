# Disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt


img_width, img_height = 224, 224

train_data_dir = 'data/train/'
test_data_dir = 'data/test/'
epochs = 1
batch_size = 16

model = applications.VGG16(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3))

for layer in model.layers:
    layer.trainable = False

print('Model loaded.')

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='sigmoid'))

model = Model(inputs= model.input, outputs= top_model(model.output))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    #rescale=1. / 255,
    #shear_range=0.2,
    #zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_generator.classes), 
        train_generator.classes)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples// batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    class_weight=class_weights)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

X_pred = model.predict_generator(train_generator, train_generator.samples // batch_size + 1)
Y_pred = model.predict_generator(test_generator, test_generator.samples // batch_size + 1)
x_pred = np.argmax(X_pred, axis=1)
y_pred = np.argmax(Y_pred, axis=1)

print('Train Confusion Matrix')
print(confusion_matrix(train_generator.classes, x_pred))
print('Classification Report')
target_names = ['Benign', 'Malign']
print(classification_report(train_generator.classes, x_pred, target_names=target_names))
print(x_pred)


print('Test Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['Benign', 'Malign']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
print(y_pred)

print('saving model...')
model.save('model.h5')