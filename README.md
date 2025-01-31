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
