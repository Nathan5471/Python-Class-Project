import tkinter as tk
import shutil
import os
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO


def draw_boxes(image, boxes, labels, scores, names):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        labelText = f"{names[int(label)]}: {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1 - 10), labelText, fill="green", font=font)

    return image


def runModel(image):
    predictions = model.predict(image)

    for prediction in predictions:
        boxes = prediction.boxes
        names = prediction.names
        for box in boxes:
            score = box.conf[0]
            image = draw_boxes(image, [box], [box.cls[0]], [score], names)

    outputImage = os.path.join("Interface/Images", "output.jpg")
    image.save(outputImage)

    return True


def loadImage(imagePath):
    image = Image.open(imagePath)
    image = image.resize((640, 640))
    return image


def uploadImage():
    imagePath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if imagePath and os.path.exists(imagePath):
        global image
        filename = os.path.basename(imagePath)
        shutil.copy(imagePath, "Interface/Images")
        imagePath = os.path.join("Interface/Images", filename)
        image = loadImage(imagePath)
        imageTk = ImageTk.PhotoImage(image)
        imageLabel.config(image=imageTk)
        imageLabel.image = imageTk  # Keep a reference to avoid garbage collection


def detectImage():
    completed = runModel(image)
    if completed:
        labeledImage = loadImage("Interface/Images/output.jpg")
        labeledImageTk = ImageTk.PhotoImage(labeledImage)
        imageLabel.config(image=labeledImageTk)
        imageLabel.image = (
            labeledImageTk  # Keep a reference to avoid garbage collection
        )


# Load the model
model = YOLO("Model/my_model.pt")

# Create the main window
window = tk.Tk()
window.title("Cat and Dog Detector")
window.attributes("-fullscreen", True)

# Create title
title = tk.Label(window, text="Cat and Dog Detector", font=("Arial", 24))
title.grid(row=0, column=0, columnspan=3, pady=20)

# Create buttons
uploadButton = tk.Button(window, text="Upload Image", command=uploadImage)
uploadButton.grid(row=1, column=0, padx=20)
detectButton = tk.Button(window, text="Detect", command=detectImage)
detectButton.grid(row=1, column=1, padx=20)
exitButton = tk.Button(window, text="Exit", command=window.quit)
exitButton.grid(row=1, column=2, padx=20)

# Create image display area
imageLabel = tk.Label(window)
imageLabel.grid(row=2, column=0, columnspan=3, pady=20)

# Run the application
window.mainloop()
