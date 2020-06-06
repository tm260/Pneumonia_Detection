# Pneumonia_Detection
In this project, I've built a machine learning model to classify Chest X-ray images of patients into Normal (Not Infected) and Pneumonia (Infected Patients).
I reached to an accuracy of 91% on the test set, which can be further improved by increasing the layers in the model or tuning the hyperparameters.
Majority of the Covid19 affected patients have developed severe pneumonia which makes it important to detect it fast and accurately in those patients, so that we can take immediate actions.
To make quick decisions, we can make use of image processing techniques along with machine learning models to detect pneumonia from the chest X-ray images.

The [dataset](https://www.kaggle.com/pcbreviglieri/pneumonia-xray-images) used in this project consists of 5863 images in total (Normal and Pneumonia). 
This dataset is an adapted version of Paul Mooney's 'Chest X-Ray Images (Pneumonia)' [dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

# Description
## Environment used
I used Google Colab with a GPU-based runntime environment to make the code execution fast. Each epoch took around 45 seconds.
## Model Architecture
The CNN model was built using Keras. It consisted of 2 Convolution layers with non linear activation funcition(ReLU). I have also added a Pooling layer(MaxPooling) after each convolution layer. At the output of the second convolutional layer we get a feature map which we pass through two fully connected layers to make predictions. 


