import csv
import cv2
import numpy as np
import sklearn

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#from sklearn.model_selection import train_test_split
#train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def getImage(source_path):
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    return image

CORRECTION = 0.2

#def generator(samples, batch_size=32):
#    num_samples = len(samples)
#    while 1:
#        #shuffle(samples)
#        for offset in range(0, num_samples, batch_size):
#            batch_samples = samples[offset : offset + batch_size]
#
#            images = []
#            measurements = []
#            for batch_sample in batch_samples:
#                image_center = getImage(batch_sample[0])
#                image_left   = getImage(batch_sample[1])
#                image_right  = getImage(batch_sample[2])
#                measurement  = float(line[3])
#
#                images.extend([image_center, image_left, image_right])
#                measurements.extend([measurement,
#                    measurement + CORRECTION,
#                    measurement - CORRECTION])
#
#                images.extend([cv2.flip(image_center, 1),
#                    cv2.flip(image_left, 1),
#                    cv2.flip(image_right, 1)])
#                measurements.extend([-measurement,
#                    -measurement - CORRECTION,
#                    -measurement + CORRECTION])
#
#            X_train = np.array(images)
#            y_train = np.array(measurement)
#            yield sklearn.utils.shuffle(X_train, y_train)

#images = []
#measurements = []
#for line in lines:
#     image_center = getImage(line[0])
#     image_left   = getImage(line[1])
#     image_right  = getImage(line[2])
#     measurement  = float(line[3])
#     images.extend([image_center, image_left, image_right])
#     measurements.extend([measurement, measurement + CORRECTION, measurement - CORRECTION])
##    for i in range(3):
##        source_path = line[i]
##        filename = source_path.split('/')[-1]
##        current_path = 'data/IMG/' + filename
##        image = cv2.imread(current_path)
##        images.append(image)
##        measurement = float(line[3])
##        measurements.append(measurement)

##X_train = np.array(images)
##y_train = np.array(measurements)

#augmented_images, augmented_measurements = [], []
#for image, measurements in zip(images, measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurements)
#    augmented_images.append(cv2.flip(image, 1))
#    augmented_measurements.append(measurement * -1.0)

#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)

images = []
measurements = []
for line in lines:
     image_center = getImage(line[0])
     image_left   = getImage(line[1])
     image_right  = getImage(line[2])
     measurement  = float(line[3])

     images.extend([
         image_center, image_left, image_right,
         cv2.flip(image_center, 1), cv2.flip(image_left, 1), cv2.flip(image_right, 1)])
     measurements.extend([
         measurement, measurement + CORRECTION, measurement - CORRECTION,
         -measurement, -measurement - CORRECTION, -measurement + CORRECTION])

X_train = np.array(images)
y_train = np.array(measurements)

#train_generator = generator(train_samples, batch_size=32)
#validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
##model.add(Flatten(input_shape=(160, 320, 3)))
#model.add(Flatten())
#model.add(Dense(1))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
#model.add(Convolution2D(6, 5, 5, activation='relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6, 5, 5, activation='relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
#model.fit_generator(train_generator,
#    steps_per_epoch=int(len(train_samples)/32),
#    validation_data=validation_generator,
#    validation_steps=len(validation_samples),
#    nb_epoch=3)

model.save('model.h5')

