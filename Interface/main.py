import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np


def combinedLoss(y_true, y_pred):
    yTrueBoxes = y_true[..., :4]
    yPredictionBoxes = y_pred[..., :4]
    yTrueClasses = y_true[..., 4]
    yPredictionClasses = y_pred[..., 4]

    boxLoss = tf.keras.losses.MeanSquaredError()(yTrueBoxes, yPredictionBoxes)

    classLoss = tf.keras.losses.BinaryCrossentropy()(yTrueClasses, yPredictionClasses)

    totalLoss = boxLoss + classLoss
    return totalLoss


model = tf.keras.models.load_model(
    "Model/catDogDetector.keras", custom_objects={"combinedLoss": combinedLoss}
)


def calculateScaleFactor(imageInfo, newWidth=244, newHeight=244):
    oldWidth = imageInfo["width"]
    oldHeight = imageInfo["height"]
    widthScaleFactor = oldWidth / newWidth
    heightScaleFactor = oldHeight / newHeight
    return (widthScaleFactor, heightScaleFactor)


def loadImage(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError(f"Image not found at path: {imagePath}")
    height, width = image.shape[:2]
    print(height, width)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (244, 244)).astype(np.float32) / 255
    print(image.shape)
    scaleFactor = calculateScaleFactor({"width": width, "height": height})
    return image, scaleFactor


def makePrediction(imagePath):
    image, scaleFactor = loadImage(imagePath)
    image = image.reshape(1, 244, 244, 3)
    prediction = model.predict(image)
    return prediction


def visualizeImage(imagePath, prediction):
    image, scaleFactor = loadImage(imagePath)
    print(scaleFactor)
    plt.imshow(image)
    ax = plt.gca()
    for box in prediction:
        x, y, width, heigth, categoryId = box
        rect = plt.Rectangle(
            (x * scaleFactor[0], y * scaleFactor[1]),
            width * scaleFactor[0],
            heigth * scaleFactor[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.show()


prediction = makePrediction(
    "C:/Users/natha/Downloads/archive/test_set/test_set/dogs/dog.4765.jpg"
)[0]
print(prediction)
visualizeImage(
    "C:/Users/natha/Downloads/archive/test_set/test_set/dogs/dog.4765.jpg", prediction
)
