# Object Detection with Python

## Introduction
This project was to train a machine learning model to accurately detect dogs and cats in images. A YOLO model from Ultralytics was used to trained the model on 8000 images (4000 cats and 4000 dogs). The trained model had an accuracy of 97% based on the training set (See more in results)

## Install and run
To install this repo you have two options
1. Clone the repo
```Bash
git clone https://github.com/Nathan5471/Python-Class-Project.git
```
2. Download as a zip

Run the following commands to install the dependencies
```Bash
pip install pillow
pip install ultralytics
```
For looking at individual images, run main.py in the Interface folder
If you want to run it on lots of images, move the images you want to run into the folder Testing/Input. Run the main.py in Testing, the outputted images will be in Testing/Output

## Example of the program running


https://github.com/user-attachments/assets/5a0b1241-abdc-4821-b9a5-7a0ecc84418a



## Results
![trainingset](https://github.com/user-attachments/assets/3521aabb-808b-476c-be05-acff1a1fe9b5)

This confusion matrix shows a 97% accuracy of detecting cats and dogs in the testing dataset of 2000 images (1000 cats and 1000 dogs). The None class was only used for when the model either detected something that isn't a cat or dog or missed a cat or dog in am image. That is why there is a 0 in None/None. 
