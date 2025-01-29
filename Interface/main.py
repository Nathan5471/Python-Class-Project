import tkinter as tk
import shutil
import os
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO


def runModel(image, filename):
    name, ext = os.path.splitext(filename)
    outputFileName = f"{name}/image0.jpg"
    model.predict(
        image,
        conf=0.4,
        save=True,
        project="Interface/Output",
        name=name,
    )
    return outputFileName


def displayThumbnails(index=0):
    for widget in thumbnailsFrame.winfo_children():
        widget.destroy()

    savedImagesPath = "Interface/Images"
    files = os.listdir(savedImagesPath)
    fileCount = len(files)
    file = -1
    small = index * 6
    large = small + 5

    for filename in files:
        if (
            filename.endswith(".jpg")
            or filename.endswith(".jpeg")
            or filename.endswith(".png")
        ):
            file += 1
            if file < small:
                continue
            if file > large:
                break
            imagePath = os.path.join(savedImagesPath, filename)
            image = loadImage(imagePath)
            image.thumbnail((100, 100))
            imageTk = ImageTk.PhotoImage(image)

            thumbnailFrame = tk.Frame(thumbnailsFrame)
            thumbnailFrame.pack(pady=5)

            thumbnailLabel = tk.Label(thumbnailFrame, image=imageTk)
            thumbnailLabel.image = imageTk
            thumbnailLabel.pack(side=tk.LEFT)

            loadButton = tk.Button(
                thumbnailFrame,
                text="Load",
                command=lambda imgPath=imagePath: loadPreviousImage(imgPath),
            )
            loadButton.pack(side=tk.RIGHT)
    if index != 0:
        previousButton = tk.Button(
            thumbnailsFrame,
            text="Previous",
            command=lambda: displayThumbnails(index - 1),
        )
        previousButton.pack(side=tk.LEFT, padx=5)
    nextIndex = index + 1
    if nextIndex * 6 < fileCount:
        nextButton = tk.Button(
            thumbnailsFrame,
            text="Next",
            command=lambda: displayThumbnails(nextIndex),
        )
        nextButton.pack(side=tk.RIGHT, padx=5)


def loadImage(imagePath, size=(640, 640)):
    image = Image.open(imagePath)
    image = image.resize(size)
    return image


def createBlankImage(size=(640, 640)):
    blank_image = Image.new("RGB", size, (255, 255, 255))
    return blank_image


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


def loadPreviousImage(imagePath):
    if imagePath and os.path.exists(imagePath):
        global image
        global filename
        filename = os.path.basename(imagePath)
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


# Creates needed folders
if not (os.path.exists("Interface/Images")):
    os.mkdir("Interface/Images")
if not (os.path.exists("Interface/Output")):
    os.mkdir("Interface/Output")

# Load the model
model = YOLO("Model/my_model.pt")

# Create the main window
window = tk.Tk()
window.title("Cat and Dog Detector")
window.attributes("-fullscreen", True)

# Create title
title = tk.Label(window, text="Cat and Dog Detector", font=("Arial", 24))
title.grid(row=0, column=0, columnspan=4, pady=20)

# Create buttons
uploadButton = tk.Button(window, text="Upload Image", command=uploadImage)
uploadButton.grid(row=1, column=0, padx=20)
loadButton = tk.Button(window, text="Load Images", command=loadPreviousImage)
loadButton.grid(row=1, column=1, padx=20)
detectButton = tk.Button(window, text="Detect", command=detectImage)
detectButton.grid(row=1, column=2, padx=20)
exitButton = tk.Button(window, text="Exit", command=window.quit)
exitButton.grid(row=1, column=3, padx=20)

# Create image display area
imageLabel = tk.Label(window)
imageLabel.grid(row=2, column=0, columnspan=4, pady=20)
blank_image = createBlankImage()
blank_imageTk = ImageTk.PhotoImage(blank_image)
imageLabel.config(image=blank_imageTk)
imageLabel.image = blank_imageTk  # Keep a reference to avoid garbage collection

# Create thumbnail display area for loaded images
thumbnailsFrame = tk.Frame(window)
thumbnailsFrame.grid(row=0, rowspan=3, column=4, padx=20, pady=20, sticky="n")
displayThumbnails()

# Run the application
window.mainloop()
