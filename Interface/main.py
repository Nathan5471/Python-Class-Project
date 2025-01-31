import tkinter as tk
import shutil
import os
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO


def runModel(image: Image, filenam: str) -> str:
    """
    Runs the YOLO model on the given image and saves the output image.
    Returns the path to the output image.

    Args:
    image: The image to run the model on.
    filename: The name of the file.

    Returns:
    The path to the output image.
    """
    name, ext = os.path.splitext(filename)
    outputFileName = f"{name}/image0.jpg"
    if os.path.exists(f"Interface/Output/{outputFileName}"):
        return outputFileName
    model.predict(
        image,
        conf=0.4,
        save=True,
        project="Interface/Output",
        name=name,
    )
    return outputFileName


def displayThumbnails(index: int = 0) -> None:
    """
    Displays thumbnails of the saved images in the thumbnailsFrame.

    Args:
    index: The index of the first image to display.

    Returns:
    None
    """
    for widget in thumbnailsFrame.winfo_children():
        widget.destroy()

    savedImagesPath = "Interface/Images"
    files = os.listdir(savedImagesPath)
    fileCount = len(files)
    file = -1
    small = index * 5
    large = small + 4

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

            deleteButton = tk.Button(
                thumbnailFrame,
                text="Delete",
                command=lambda imgPath=imagePath: deletedSavedImage(imgPath),
            )
            deleteButton.pack(side=tk.RIGHT)
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


def loadImage(imagePath: str, size: tuple[int, int] = (640, 640)) -> Image:
    """
    Loads an image from the given path and resizes it to the given size.

    Args:
    imagePath: The path to the image.
    size: The size to resize the image to. Default is (640, 640).

    Returns:
    The loaded and resized image.
    """
    image = Image.open(imagePath)
    image = image.resize(size)
    return image


def resizeImage(image: Image, size: tuple[int, int] = (640, 640)) -> Image:
    """
    Resizes the given image to the given size.

    Args:
    image: The image to resize.
    size: The size to resize the image to. Default is (640, 640).

    Returns:
    The resized image.
    """
    return image.resize(size)


def deletedSavedImage(imagePath: str) -> None:
    """
    Deletes the image at the given path.

    Args:
    imagePath: The path to the image.

    Returns:
    None
    """
    os.remove(imagePath)
    displayThumbnails()


def createBlankImage(size: tuple[int, int] = (640, 640)) -> Image:
    """
    Creates a blank image with the given size.

    Args:
    size: The size of the image. Default is (640, 640).

    Returns:
    The blank image.
    """
    blank_image = Image.new("RGB", size, (255, 255, 255))
    return blank_image


def uploadImage() -> None:
    """
    Opens a file dialog to upload an image and displays it in the imageLabel.

    Args:
    None

    Returns:
    None
    """
    global imagePath
    imagePath = None
    imagePath = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not (imagePath and os.path.exists(imagePath)):
        tk.messagebox.showerror("Error", f"Invalid file path: {imagePath}")
    global image
    global filename
    filename = os.path.basename(imagePath)
    imagePath = os.path.join("Interface/Images", filename)
    if os.path.exists(imagePath):
        tk.messagebox.showerror(
            "Error", "Image already exists, loading it from saved images."
        )
        loadPreviousImage(imagePath)
    else:
        shutil.copy(imagePath, "Interface/Images")
        image = loadImage(imagePath)
        displayImage(image)
        displayThumbnails()


def saveImage(image: Image) -> None:
    """
    Saves the given image to the saved images.

    Args:
    image: The image to save.

    Returns:
    None
    """
    if not image:
        tk.messagebox.showerror("Error", "No image to save.")
        return
    imagePath = filedialog.asksaveasfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if imagePath:
        try:
            image.save(imagePath)
            tk.messagebox.showinfo("Success", "Image saved successfully.")
        except:
            tk.messagebox.showerror(
                "Error",
                "Failed to save the image. Please try again and make sure to use a file extension of .jpg, .jpeg, or .png.",
            )


def loadPreviousImage(imagePath: str) -> None:
    """
    Loads an image from the saved images based on the given path and displays it in the imageLabel.

    Args:
    imagePath: The path to the image.

    Returns:
    None
    """
    if imagePath and os.path.exists(imagePath):
        global image
        global filename
        filename = os.path.basename(imagePath)
        image = loadImage(imagePath)
        displayImage(image)
    else:
        tk.messagebox.showerror("Error", f"Invalid file path: {imagePath}")


def detectImage() -> None:
    """
    Runs the YOLO model on the uploaded image and displays the labeled image in the imageLabel.

    Args:
    None

    Returns:
    None
    """
    global image
    outputFileName = runModel(image, filename)
    if outputFileName:
        image = loadImage(f"Interface/Output/{outputFileName}")
        displayImage(image)
    else:
        tk.messagebox.showerror("Error", "Failed to run the model.")


def clearImage() -> None:
    """
    Clears the uploaded image from the imageLabel.

    Args:
    None

    Returns:
    None
    """
    global image
    global filename
    image = createBlankImage()
    displayImage(image)
    image = None  # Clear the image
    filename = None


def displayImage(image: Image) -> None:
    """
    Displays the given image in the imageLabel.

    Args:
    image: The image to display.

    Returns:
    None
    """
    image = resizeImage(image, (500, 500))
    imageTk = ImageTk.PhotoImage(image)
    imageLabel.config(image=imageTk)
    imageLabel.image = imageTk  # Keep a reference to avoid garbage collection


# Creates needed folders
if not (os.path.exists("Interface/Images")):
    os.mkdir("Interface/Images")
if not (os.path.exists("Interface/Output")):
    os.mkdir("Interface/Output")

# Load the model
if not (os.path.exists("Model/my_model.pt")):
    print("Model not found.")
    print("Please download the model from the following link:")
    print("https://github.com/Nathan5471/Python-Class-Project")
    exit()
model = YOLO("Model/my_model.pt")

# Create the main window
window = tk.Tk()
window.title("Cat and Dog Detector")
window.attributes("-fullscreen", True)

# Create title
title = tk.Label(window, text="Cat and Dog Detector", font=("Arial", 24))
title.grid(row=0, column=0, columnspan=5, pady=20)

# Create buttons
uploadButton = tk.Button(window, text="Upload Image", command=uploadImage)
uploadButton.grid(row=1, column=0, padx=20)
detectButton = tk.Button(window, text="Detect", command=detectImage)
detectButton.grid(row=1, column=1, padx=20)
clearButton = tk.Button(window, text="Clear", command=clearImage)
clearButton.grid(row=1, column=2, padx=20)
saveButton = tk.Button(window, text="Save", command=lambda: saveImage(image))
saveButton.grid(row=1, column=3, padx=20)
exitButton = tk.Button(window, text="Exit", command=window.quit)
exitButton.grid(row=1, column=4, padx=20)

# Create image display area
imageLabel = tk.Label(window)
imageLabel.grid(row=2, column=0, columnspan=5, pady=20)
blankImage = createBlankImage()
displayImage(blankImage)

# Create thumbnail display area for loaded images
thumbnailsFrame = tk.Frame(window)
thumbnailsFrame.grid(row=0, rowspan=3, column=5, padx=20, pady=20, sticky="n")
displayThumbnails()

# Run the application
window.mainloop()
