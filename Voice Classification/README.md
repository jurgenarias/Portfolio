
<img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.crunchbase.com%2Forganization%2Fgeneral-assembly&psig=AOvVaw0j4FTphH6c3vOhKXf1aRI0&ust=1600049700266000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCMjs2-OH5esCFQAAAAAdAAAAABAP" width="100" height="100" alt="GA" img align="right">

# VOICE CLASSIFICATION

<img src="https://cdn.ttgtmedia.com/rms/onlineImages/mobile_computing-mobile%20biometrics_05.png" width="800" height="400" alt="voice" img align="center">
Project by <a href="https://www.linkedin.com/in/jurgen-arias-02371117/" rel="nofollow">Jurgen Arias</a>.

<br></br>


## Contents

- [Problem Statement](#Problem-Statement)
- [Summary of Project](#Summary-of-Project)
- [Urban Sound Challenge](#Urban-Sound-Challenge)
- [Speaker Classifier](#Speaker-Classifier)
- [Speaker's Gender Classifier](#Speaker's-Gender-Classifier)
- [Future Explorations](#Future-Explorations)
- [Sources](#Sources)

<br></br>

## Problem Statement

Imagine a conference room, a meeting is taking place. There are several people, they all take turns to speak. We want to transcribe what they are saying. There are tools that can transcribe voice to text (most of us have them on our phones already), but can we train a model to learn voices and accurately predict who is speaking just by listening to their voice? Can we predict their gender?

Imagine a conference room, a meeting is taking place. There are several people, they all take turns to speak. We want to transcribe what they are saying. There are tools that can transcribe voice to text (most of us have them on our phones), but can we train a model to learn voices and accurately predict who is speaking just by listening to their voice? Can we predict their gender?

We will tackle these problems by using Machine Learning with Neural Networks.

<br></br>

## Summary of Project

The first problem was to learn how to manipulate audio data and build models to classify sounds. We found a great competition called the Urban Sound at https://datahack.analyticsvidhya.com/contest/practice-problem-urban-sound-classification/. The problem is to classify 10 different urban sounds like children playing, street music, a car engine, etc. We did a lot of research to understand how to solve the problem and how a lot of people have tackled the problem. We focused on two approaches. The first approach was to extract numerical features from the audio clips using the librosa library from python and using those features to train a neural network model (NN) and the second approach was to convert the audio clips to pictures and use those images to train a convolutional neural network model (CNN).

We got good results with the NN (93% on test data) and with the CNN (92% on test data). We combined those two models together in a voting classifier by joining the probability of the predictions and got a 95% accuracy when using the NN and CNN together.

Now we could go forward and use what we learned to tackle the person and gender classifier problem.

We used an NN model for predicting a classification among 115 speakers and got 99.8% accuracy. We did not do the CNN for this because of the high accuracy (almost perfect) of our NN model.

We used an NN model for predicting gender and got 99.8% accuracy when classifying the gender of speakers that the model had listened to before. We got new data from speakers that the model had never heard before and got a 95% accuracy.

<br></br>

## Urban Sound Challenge:

#### Feed Forward Neural Network

We got the link for the data from the competition website. They are in a google drive at: https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU. We only used the train data since we wanted to test on label data to see how good our model was. The test data is only for predictions for the competition.

The data contains 5435 labeled sounds from 10 different classes. The classes are siren, street music, drilling, engine idling, air conditioner, car horn, dog bark, drilling, gun shot and jackhammer. Most classes are balanced but there are two that have low representation. Most represent 11% of the data but one only represents 5% and one only 4%. We did not balanced classes because we took it as a good challenge to build a good model with somewhat unbalanced classes.

Librosa is a fantastic library for python to use with audio files and it is what most people used in the audio classification problems. We used the librosa library to extract features. After doing some research, we found some features from the librosa information page at: https://librosa.github.io/librosa/0.6.0/feature.html.

We loaded the csv file that came with the training data into a dataframe with all the names of the audio files and its corresponding label. We extracted the features through a function that iterates through every row of the dataframe accessing the file in the computer by reading the file's path. We used Mel-frequency Cepstral Coefficients (MFCCs), Chromagram, Mel-scaled Spectrogram, Spectral Contrast and Tonal Centroid Features (tonnetz). We got an array of 193 features with their respective label. We set them to be our X and y. We split them into training, validation and test data, checked that we kept the same proportions for the classes as our total data and scaled the data. We chose 3435 audios for our train data, 1000 for our validation data and 1000 for our test data.

We built a feed forward neural network with dense layers with two hidden layers using relu and softmax for the 10 outputs. We compiled the model using the adam optimizer and categorical crossentropy for loss. We gridsearched the best parameters for the number of neurons and the dropout proportions for our layers and came up with a decent model that predicted our test never before seen (or heard) data with an accuracy of 93%.

#### Convolutional Neural Network

Using the same data and the same steps as above, we generated our dataframe for our audio files. Now we needed to use a function to create images for every audio file. As before, our function iterated through every row of the dataframe and created an image using librosa and saved it to a local folder. Here is the information about how to create an image from librosa and the different kinds of images you can create: https://librosa.github.io/librosa/generated/librosa.display.specshow.html.

After creating the images, we again split the our data intro training, validation and test (we used the same proportions as with the neural network from before). We checked that we have the same balance on classes and changed our dataframe file names from .wav to .jpg so that we could use the same dataframe to access the images from our local folder.

Keras has some wonderful information about how to load and preprocess images from your local files specially the .flow_from_dataframe which is what we used. The documentation is here: https://keras.io/preprocessing/image/. This way we loaded our train, validation and test data into generators and we are ready to build the model.

We built a convolutional neural network with a Conv2D and MaxPooling2D input and five hidden layers: three Conv2D with their respective MaxPooling2D, then flatten and two Dense layers (all with relu). Finally we had the Dense output layer for the 10 classes with softmax activation. We again, compiled the model using the adam optimizer and categorical crossentropy for loss. We did not use gridsearch because it took too long (about two hours) so we trained on 250 epochs. We got 92% accuracy on our never before seen test data.

#### Voting Classifier

We decided that since we have two models that do the same thing, we might as well use them together. We got the predictions probabilities for each class from our NN and the predictions probabilities for our CNN and added them together and got the maximum for each one. In other words, if my NN was 65% sure that a sound was some children playing and my CNN was 95% sure it was instead street music, then street music would have a higher probability thus my prediction would have to be street music. We did this and bumped our predictions to be 95% accurate on never before seen test data. I found this to be an amazing way to combine different models together to have better predictions.

<br></br>

## Speaker Classifier

Now we have the tools to tackle our original problem which was classifying speakers. First problem was to get good audio data. After much research, we run into an absolutely amazing database for audio clips from audiobooks recordings at: http://www.openslr.org/12/. This dataset contains many gigabytes of clean data in ".flac" files that work great with macs. We used a subset of the train-clean-100.tar.gz [6.3G]. The data comes very well organized in folders by speakers with speaker ids, books, chapters and file number. We also get a ".txt" file with information from the speakers letting us know their gender, how long are their recordings and even their name. The subset we used has 13,000 voice clips usually ranging from 12 to 18 seconds.

We tried the feed forward neural network since it is faster and it gave us better accuracy for our Urban Challenge problem. We followed the same steps as before now dealing with much more data (around 13k instead of 5k) and longer voice clips (average of around 14 seconds instead of 4 seconds). Extracting the features took around 3 hours but it has to be done only once and we can save that array as a numpy array in a local folder and load it whenever we need to use it or change anything.

We used 115 different speakers both male and female where the minimum number of voice clips per speaker was 56 and the maximum was 166 (randomly selected). The standard deviation of the number of clips per speaker was about 16. We can consider them to be balanced classes.

We fitted the data into a neural network model with the same configuration of our gridsearched model from the Urban Challenge and got a whopping 99.8% accuracy. We predicting on 1312 audio samples and classified them into the 115 speakers and only got two audio samples wrong. The model only took 20 seconds to fit and it was almost perfect so we decided it was not necessary to do the CNN model.

<br></br>

## Speaker's Gender Classifier

Using the same amount of data as for the speaker classifier, we labeled the voice clips by male and female by placing them into two separate folders. We generated dataframes from each folder and then concatenated (fancy term for putting together) them into a dataframe. We shuffled the data and reset the index to get a brand new dataframe of all the files with labeled gender data.

We used the same split and extracted the same features as our Urban Challenge NN model. We used the same configuration for our Feed Forward Neural Network and again, it only took 20 seconds to fit the model. We got an accuracy of 99.8% on our test data. Even though our model had never seen (or heard) our test data, it has been trained with data that contains the same people speaking, that is why our model is almost perfect. We taught the model which speaker was male and female and when we predicted new voice clips, it was almost perfect because it already knew the people. After considering this, we gathered more data. We collected 100 new voice clips from new people that our model had never heard before. We cleaned it in the same way and made predictions on it. We got 97% accuracy on our new test data from 100 never before heard speakers.

We have a good model but it is not almost perfect like the model for Speaker Classifier so we decided to do a CNN to see if we could improve our results.

Just like in the urban classifier problem, we created images from our data that we labeled and placed into local folders. We created the dataframes and used our function to create the images in a different folder to use with the Keras Generator. Since we are dealing with voice, we reduced the frequency of our audio clip images to only include frequencies from 50 Hz to 280 Hz. "The voiced speech of a typical adult male will have a fundamental frequency from 85 to 180 Hz, and that of a typical adult female from 165 to 255 Hz" (from https://en.wikipedia.org/wiki/Voice_frequency). We again used the same CNN configuration as before and fitted the model (which took a couple of hours). We got 97.7% accuracy on our test data. Remember we got 99.8% accuracy on our test data with our simple dense feed forward neural network so this was a little disappointing. We again, generated predictions on 100 new never before heard speakers and got 95% accuracy.

We could combine these two models with a voting classifier and get a better accuracy or train the models with more data to make them more accurate but we have a time constraint and want to implement this model to be able to use it for an interactive demonstration and the CNN model's process takes too long. It takes a long time to create the images and also to fit the model so we will use the NN with 97% accuracy since it is fast and accurate.

## Future Explorations:

- We would like to gridsearch over the best parameters on our CNN and see if we could get the same or better accuracy than our Dense Layered Neural Network. This could also be done by uploading the data and using Google Colab since they provide free GPU usage that make Neural Networks run considerably much faster. If your laptop takes 4 hours to fit a neural network, Google Colab can probably do it in 15 minutes. If Google Colab sounds interesting, I recommend reading this blog post from my friend and colleague Brenda Hali about Google Colab: https://towardsdatascience.com/google-colab-jupyter-lab-on-steroids-perfect-for-deep-learning-cdddc174d77a.

- We would like to try CNN models with all the different kinds of images that librosa provides from audio files and see which kind of images give better predictions.

- We would like to add more training data to see if we can get better results in the speaker's gender classifier. We had to label the data by hand and this was time consuming. With more data we can probably have more accurate models.

- We would also like to fit a Recurrent Neural Network and see how accurate it is since they are good with time series data and voice clips are basically time series.

<br></br>


## Sources:

Datasets:

http://www.openslr.org/12/

https://drive.google.com/drive/folders/0By0bAi7hOBAFUHVXd1JCN3MwTEU.

Articles and helpful links:

1) https://medium.com/@patrickbfuller/librosa-a-python-audio-libary-60014eeaccfb

2) https://www.hindawi.com/journals/sp/2019/7213717/

3) https://www.reddit.com/r/programming/comments/4pb03r/identifying_the_gender_of_a_voice_using_machine/

4) https://towardsdatascience.com/deep-learning-tips-and-tricks-1ef708ec5f53

5) https://librosa.github.io/librosa/0.6.0/feature.html

6) https://medium.com/@CVxTz/audio-classification-a-convolutional-neural-network-approach-b0a4fce8f6c

7) https://github.com/CVxTz/audio_classification/blob/master/code/keras_cnn_mel.py

8) https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/

10) https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7

11) https://github.com/mikesmales/Udacity-ML-Capstone/blob/master/Notebooks/4%20Model%20Refinement.ipynb

12) http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/

13) https://upcommons.upc.edu/bitstream/handle/2117/86673/113166.pdf

14) http://www.primaryobjects.com/2016/06/22/identifying-the-gender-of-a-voice-using-machine-learning/

15) https://www.endpoint.com/blog/2019/01/08/speech-recognition-with-tensorflow

16) https://towardsdatascience.com/how-to-build-a-speech-recognition-bot-with-python-81d0fe3cea9a

17) https://github.com/scaomath/UCI-Math10/blob/master/Lectures/Lecture-22-Neural-Network-I.ipynb

18) https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4

19) https://librosa.github.io/librosa/generated/librosa.display.specshow.html

20) https://keras.io/preprocessing/image/

21) https://github.com/sainathadapa/kaggle-freesound-audio-tagging

22) https://keras.io/models/sequential/
