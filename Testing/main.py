import os
from ultralytics import YOLO
import cv2

model = YOLO("Model/my_model.pt")
inputFolder = "Testing/Input"
if not (os.path.exists(inputFolder)):
    os.mkdir(inputFolder)
outputFolder = "Testing/Output"
if not (os.path.exists(outputFolder)):
    os.mkdir(outputFolder)

print("Labeling images...")
inputFiles = os.listdir(inputFolder)
inputFilesAmount = len(inputFiles)
print("{inputFilesAmount} images found in the input folder.")
currentInputFile = 0
results = model.predict(source=inputFolder, save=True, save_dir=outputFolder)

print("Image labeling complete!")
