import tkinter as tk
import shutil
import os
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO


def runModel(image, filename):
    model.predict(source=imagePath, conf=0.4, save=True, save_dir="Interface/Images")

    name, ext = os.path.splitext(filename)
    outputFileName = f"{name}_predictions{ext}"

    return outputFileName


def loadImage(imagePath):
    image = Image.open(imagePath)
    image = image.resize((640, 640))
    return image


def uploadImage():
    global imagePath
    imagePath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if imagePath and os.path.exists(imagePath):
        global image
        global filename
        filename = os.path.basename(imagePath)
        shutil.copy(imagePath, "Interface/Images")
        imagePath = os.path.join("Interface/Images", filename)
        image = loadImage(imagePath)
        imageTk = ImageTk.PhotoImage(image)
        imageLabel.config(image=imageTk)
        imageLabel.image = imageTk  # Keep a reference to avoid garbage collection


def detectImage():
    outputFileName = runModel(image, filename)
    if outputFileName:
        labeledImage = loadImage(f"Interface/Output/{outputFileName}")
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
