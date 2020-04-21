# imports
import os
import numpy as np
import pandas as pd
from keras import models


# consts
SIZE = 28

# paths
# train_path = 'train.csv'
test_path = 'test.csv'
models_path = 'models/'

# get data
# train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# get image data
test_x = np.uint8(test_data.drop(['id'], axis=1)).reshape(test_data.shape[0], SIZE, SIZE, 1)

# convert to float
test_x = test_x / 255.0

# get id
test_id = test_data['id']

# free space
# del train_data
del test_data

# load model structures
with open(os.path.join(models_path, 'model.json'), 'r') as f:
    model = models.model_from_json(f.read())

# load model weights
model.load_weights(os.path.join(models_path, 'model.h5'))

# make predictions
pred = model.predict(test_x)

# summarize predictions
results = []
for idx in range(len(pred)):
    results.append(int(pred[idx].argmax()))

# prepare predictions df
submission = pd.DataFrame(
    {'id': test_id,
     'label': results}
                         )

# save predictions to file
submission.to_csv('submission.csv', index=False)
