import os
from ultralytics import YOLO
import cv2

model = YOLO("Model/my_model.pt")
inputFolder = "Testing/Input"
outputFolder = "Testing/Output"

print("Labeling images...")
inputFiles = os.listdir(inputFolder)
inputFilesAmount = len(inputFiles)
print("{inputFilesAmount} images found in the input folder.")
currentInputFile = 0
for filename in inputFiles:
    currentInputFile += 1
    if currentInputFile % 10 == 0:
        print(f"Labeling image {currentInputFile} of {inputFilesAmount}")
    if filename.endswith((".jpg", ".jpeg", ".png")):
        imagePath = os.path.join(inputFolder, filename)
        image = cv2.imread(imagePath)

        results = model.predict(image)

        for result in results:
            boxes = result.boxes
            names = result.names
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = names[int(box.cls[0])]
                score = box.conf[0]
                labelText = f"{label}: {score:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    labelText,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        output_path = os.path.join(outputFolder, filename)
        cv2.imwrite(output_path, image)

print("Image labeling complete!")
