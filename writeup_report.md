#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/track1.jpg "Driving on Track 1"
[image2]: ./examples/track1_counter_sense.jpg "Driving on Track 1 - Counter Sense"
[image3]: ./examples/track2.jpg "Driving on Track 2"
[image4]: ./examples/track2_counter_sense.jpg "Driving on Track 2 - Counter Sense"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **model.h5** containing a trained convolution neural network
* **video.mp4** containig the video of one run for track one
* **video_track2.mp4** containing the video of one lap for track two
* **writeup_report.md** summarizing the results
* **video.py** for generating the video from a serie of images (already included in the project template, but modified to be compatible with python 3.5.2)
* **drive.py** for driving the car in autonomous mode (already included in the project's template and not modified)
* **bak/** a directory containing scripts, models and videos of previous runs (just for curiosity)
* **data/** a directory containing the images and drive log to train and verify the model (just for curiosity)

####2. Submission includes functional code
Using the Udacity provided simulator and my **drive.py** file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Note: In order to develop this project I used a GPU instance set up from the scratch wich uses python 3.5.2, tensorflow 1.3.0,  keras 2.0.8, cuda 8.0.61 and cudnn 6.0.21, so the driving environment had to be set up to be compatible with these versions, specially using the same versions of keras and python. 

####3. Submission code is usable and readable

The **model.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Based on the architecture presented during the lessons, my model consists of a convolution neural network with the following layers (model.py lines 71-84)

* Convolutional layer: 5x5 filter size, deep of  24, strides of 2x2,  with RELU activation
* Convolutional layer: 5x5 filter size, deep of  36, strides of 2x2,  with RELU activation
* Convolutional layer: 5x5 filter size, deep of  48, strides of 2x2,  with RELU activation
* Convolutional layer: 3x3 filter size, deep of  64, strides of 1x1,  with RELU activation
* Convolutional layer: 3x3 filter size, deep of  64, strides of 1x1,  with RELU activation
* Flattening layer
* Dense layer with 100 outputs and RELU activation
* Dense layer with 50 outputs and RELU activation
* Dense layer with 10 outputs and RELU activation
* Dense layer with 1 output and RELU activation

The model includes RELU layers to introduce nonlinearity, the data is normalized in the model using a Keras lambda layer (code line 72), and the images are cropped to remove the upper horizon and hood parts (code line 73) . 

####2. Attempts to reduce overfitting in the model

The model contains a dropout layer with a rate of 50% during the input in order to reduce overfitting (model.py line 24).

Te model also was trainned in batches of size 32, using for that purpose the method to generate them (code lines 49-63) and provided to the Kera's fit generator (code line 112). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 18-46), including samples of-track one and track-two, driving in counter sense, and the images were also flipped. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in the opposite sense and driving samples on both tracks.

For details about how I created the training data, you can look at the directory **data/**. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one exposed during the lessons, I thought this model might be appropriate because it was showing progressive improvements during the evolution of the examples.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (20% of the total).

A problem I was facing was that during the tests was that the car was exiting on the non-paved roads, which might sugests that the model was generalizing too much to take any drivable surface, that is because I was using drop-out in every layer, I decided to reduce the drop-out only to the input layer to avoid over generalization, that produced good results.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road, in the case of the track-two the model was doing a better job than me on the manual traning.

####2. Final Model Architecture

The final model architecture (model.py lines 71-84) consisted of a convolution neural network with the following layers and layer sizes:

* Convolutional layer: 5x5 filter size, deep of  24, strides of 2x2,  with RELU activation
* Convolutional layer: 5x5 filter size, deep of  36, strides of 2x2,  with RELU activation
* Convolutional layer: 5x5 filter size, deep of  48, strides of 2x2,  with RELU activation
* Convolutional layer: 3x3 filter size, deep of  64, strides of 1x1,  with RELU activation
* Convolutional layer: 3x3 filter size, deep of  64, strides of 1x1,  with RELU activation
* Flattening layer
* Dense layer with 100 outputs and RELU activation
* Dense layer with 50 outputs and RELU activation
* Dense layer with 10 outputs and RELU activation
* Dense layer with 1 output and RELU activation


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a few laps on track one first trying to stay on the center of the lane, then trying to recover if I was to close to the right or left border. Here is an example image of driving in track one:

![alt text][image1]

Then, I recorded the vehicle driving also in track one, but in counter sense, here is an example:

![alt text][image2]

Then I recorder few laps on track two two in the original sense and two in counter sense, it was actually very ambitious to drive in the center of a lane due to the difficulty of the track (using the mouse was almost impossible, I had to use a joystick), so I just was trying to drive staying in one lane, probably the training in track2 was better to the model for learning to recover, these are some examples:

![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would help the model to generalize in different roads and to stay in one lane.

After the collection process, I had 113,580 number of data points. I then preprocessed this data by scaling the color to be each one in the range of [-0.5, 0.5]. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the loss rate, which was not improving with more epochs, I used an adam optimizer so that manually training the learning rate wasn't necessary.

###Track Two

The final model was able to drive and stay on the road for track two, in fact it was doing a better job than me trying to do the manual training. I think track 2 provided a better way to train the model to recover and stay on track, because pretty much is what I was trying to do naturally. I recorded a video of a lap driving in Track Two, which can be watch in **video_track2.mp4**.
