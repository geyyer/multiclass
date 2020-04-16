# imports
import numpy as np
import os
import pandas as pd
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from model import ModelCustomInception
from sklearn.model_selection import train_test_split

# consts
SIZE = 28
TEST_FRAC = 0.2
RANDOM_SEED = 42

# paths
base_path = os.getcwd()
data_path = os.path.join(base_path, 'data')
train_path = os.path.join(data_path, 'train.csv')
# test_path = os.path.join(data_path, 'test.csv')

# get data
train_data = pd.read_csv(train_path)
# test_data = pd.read_csv(test_path)

# prep data
data_x = np.uint8(train_data.drop(['label'], axis=1)).reshape(train_data.shape[0], SIZE, SIZE, 1)
# convert to float
data_x = data_x / 255.0

# prep labels
data_y = train_data['label']
data_y = to_categorical(data_y, num_classes=data_y[0].shape)

# clean up
del train_data
# del test_data

train_x, val_x, train_y, val_y = train_test_split(data_x,
                                                  data_y,
                                                  test_size=TEST_FRAC,
                                                  random_state=RANDOM_SEED
                                                  )

# get data shape
src_size = (train_x.shape[1], train_x.shape[2], 1)
dst_size = (train_y.shape[1])

# instantiate a model
model = ModelCustomInception(src_size=src_size,
                             dst_size=dst_size
                             )

# build a model
model.build(optimizer='Adam',
            lr=1e-3
            )

# model summary
model.selfie()

epochs = 50
batch_size = 32

# data augmentation
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=20,
                             zoom_range=0.2,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=False,
                             vertical_flip=False
                             )
datagen.fit(train_x)

# train the model
model.train_model_holdout(train_x=train_x,
                          train_y=train_y,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(val_x, val_y),
                          data_size=-1
                          )

# train set evaluation
print('evaluating train set')
results = model.model.evaluate(train_x, train_y)
print('train mse, train mae:', results)

# test set evaluation (never used in training)
print('evaluating test set')
results = model.model.evaluate(val_x, val_y)
print('test mse, test mae:', results)

# save the model
model.save()
