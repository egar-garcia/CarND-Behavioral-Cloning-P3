import csv
import cv2
import numpy as np
import sklearn
from math import ceil
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Dropout, Activation
from keras.layers import Cropping2D, Lambda


IMAGES_PATH = 'data/IMG/'
CORRECTION  = 0.2

def getImage(source_path):
    """
    Returns an image from the path retrieved from the driving log of the
    training mode.
    """
    filename = source_path.split('/')[-1]
    current_path = IMAGES_PATH + filename
    image = cv2.imread(current_path)
    return image


def addImagesAndAngles(imagesSet, anglesSet, sample):
    """
    Adds the images and the steering angles to the given (training and/or validation) sets
    """
    # Getting center image, right image, left images and steering angle
    image_center = getImage(sample[0])
    image_left   = getImage(sample[1])
    image_right  = getImage(sample[2])
    angle        = float(sample[3])

    # Adding the images to the images' set
    imagesSet.extend([image_center, image_left, image_right])
    # Adding steering angles to the angles' set, doing a correction for left and right images
    anglesSet.extend([angle, angle + CORRECTION, angle - CORRECTION])

    # Adding flipped images to extend images' and angles' sets
    imagesSet.extend([cv2.flip(image_center, 1), cv2.flip(image_left, 1), cv2.flip(image_right, 1)])
    anglesSet.extend([-angle, -angle - CORRECTION, -angle + CORRECTION])


def set_generator(samples, batch_size = 32):
    """
    Generator to process the samples in batches during the training
    """
    num_samples = len(samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            images = []
            angles = []
            for sample in batch_samples:
                addImagesAndAngles(images, angles, sample)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train


# Neural Network Model

BATCH_SIZE   = 32;
DROPOUT_RATE = 0.33;

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Dropout(DROPOUT_RATE))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


# Getting the information of the samples from a CSV file resulting from
# the recording of the manual driving in the 'Training Model' of the simulator
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)


# Getting the training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
train_generator      = set_generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = set_generator(validation_samples, batch_size=BATCH_SIZE)
print('Number of recorded samples: ', len(samples))
print('Training set size: ', len(train_samples), ', validation set size: ', len(validation_samples))


print('Starting trainning ...')

model.fit_generator(train_generator,
                    int(ceil(len(train_samples) / BATCH_SIZE)),
                    epochs = 5,
                    validation_data = validation_generator,
                    validation_steps = int(ceil(len(validation_samples) / BATCH_SIZE)))
model.save('model.h5')

print('Done.')
