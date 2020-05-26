from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras.models import Model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys


# path to the model weights files.
#weights_path = '../keras/examples/vgg16_weights.h5'
# top_model_weights_path = 'models/bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'
nb_train_samples = 900
nb_validation_samples = 379
epochs = 20
batch_size = 2

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False,
                           input_shape=(128, 128, 3))
print('Model loaded.')

for layer in model.layers:
    layer.trainable = False

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

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
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
best_model_VA = ModelCheckpoint('BM_VA_ex1_model',monitor='val_acc',
                                mode = 'max', verbose=1, save_best_only=True)
best_model_VL = ModelCheckpoint('BM_VL_ex1_model',monitor='val_loss',
                                mode = 'min', verbose=1, save_best_only=True)


model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples, callbacks=[best_model_VA,best_model_VL])

print('saving model...')
model.save('ex1_model.h5')