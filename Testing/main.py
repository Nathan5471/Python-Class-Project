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
results = model.predict(
    source=inputFolder, save=True, project="Testing/Output", name="Results"
)

print("Image labeling complete!")
