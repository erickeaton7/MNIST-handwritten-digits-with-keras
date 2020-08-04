import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Read csvs
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# convert to np arrays
train_array = np.array(train)
test_images = np.array(test)

# separate images from labels
train_images = [img[1:] for img in train_array]
train_labels = [img[0] for img in train_array]

# Normalize pixel values for network
train_images = [((img/255) - 0.5) for img in train_images]
test_images = [((img/255) - 0.5) for img in test_images]

# 2 layers with 64 neurons using relu
# 1 layer with 20 neurons using softmax
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit() takes np arrays
train_images = np.array(train_images)
test_images = np.array(test_images)

# train the model
model.fit(train_images, to_categorical(train_labels), epochs=10, batch_size=32)

# find model predictions on test images
predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=1)
image_ids = range(1,28001)

# put results into submission format
submission_df = pd.DataFrame(list(zip(image_ids, predictions)), columns = ['ImageId', 'Label'])

# save as csv
submission_df.to_csv('submission.csv', index=False)
