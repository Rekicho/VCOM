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


# path to the model weights files.
# weights_path = '../keras/examples/vgg16_weights.h5'
# top_model_weights_path = 'models/bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'
nb_train_samples = 900
nb_validation_samples = 379
epochs = 2
batch_size = 16

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3))
print('Model loaded.')

for layer in model.layers:
    layer.trainable = False

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
#model.add(top_model)
model = Model(inputs= model.input, outputs= top_model(model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
#for layer in model.layers[:25]:
#    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    #rescale=1. / 255,
    #shear_range=0.2,
    #zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_generator.classes), 
        train_generator.classes)

# fine-tune the model
best_model_VA = ModelCheckpoint('task1_model',monitor='val_acc',
                                mode = 'max', verbose=1, save_best_only=True)
best_model_VL = ModelCheckpoint('task1_model',monitor='val_loss',
                                mode = 'min', verbose=1, save_best_only=True)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[best_model_VA,best_model_VL])
    ,class_weight=class_weights)

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


X_pred = model.predict_generator(train_generator, nb_train_samples // batch_size + 1)
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size + 1)
x_pred = np.argmax(X_pred, axis=1)
y_pred = np.argmax(Y_pred, axis=1)

print('Train Confusion Matrix')
print(confusion_matrix(train_generator.classes, x_pred))
print('Classification Report')
target_names = ['Benign', 'Malign']
print(classification_report(train_generator.classes, x_pred, target_names=target_names))
print(x_pred)


print('Test Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Benign', 'Malign']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
print(y_pred)

print('saving model...')
model.save('model.h5')