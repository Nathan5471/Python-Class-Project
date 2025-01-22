import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model("Model/catDogDetector.keras")


def loadImage(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError(f"Image not found at path: {imagePath}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (244, 244)).astype(np.float32) / 255
    return image


def makePrediction(imagePath):
    image = loadImage(imagePath)
    image = image.reshape(1, 244, 244, 3)
    prediction = model.predict(image)
    return prediction


def visualizeImage(imagePath, prediction):
    image = loadImage(imagePath)
    plt.imshow(image)
    ax = plt.gca()
    for box in prediction:
        x, y, width, heigth, categoryId = box
        rect = plt.Rectangle(
            (x, y),
            width,
            heigth,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.show()


prediction = makePrediction(
    "C:/Users/natha/Downloads/archive/test_set/test_set/dogs/dog.4666.jpg"
)[0]
print(prediction)
visualizeImage(
    "C:/Users/natha/Downloads/archive/test_set/test_set/dogs/dog.4666.jpg", prediction
)
