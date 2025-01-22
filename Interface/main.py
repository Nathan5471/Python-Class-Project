import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("Model/catDogDetector.h5")


def loadImage(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError(f"Image not found at path: {imagePath}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (244, 244))
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
        x, y, width, heigth = box
        rect = plt.Rectangle(
            (x, y),
            (width * 244),
            (heigth * 244),
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.show()


prediction = makePrediction(
    "C:/Users/natha/Downloads/archive/test_set/test_set/dogs/dog.4989.jpg"
)[0]
print(prediction)
visualizeImage(
    "C:/Users/natha/Downloads/archive/test_set/test_set/dogs/dog.4846.jpg", prediction
)
