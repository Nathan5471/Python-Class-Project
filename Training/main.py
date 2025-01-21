import json
import tensorflow as tf
from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image

# Load the COCO dataset
catAnnotationFile = "Training Data/Training Set/Cats/cats.json"
dogAnnotationFile = "Training Data/Training Set/Dogs/dogs.json"
catCOCO = COCO(catAnnotationFile)
dogCOCO = COCO(dogAnnotationFile)

catImageIds = list(catCOCO.imgs.keys())
dogImageIds = list(dogCOCO.imgs.keys())


# Load the images
def calculateScaleFactor(imageInfo, newWidth=244, newHeight=244):
    oldWidth = imageInfo["width"]
    oldHeight = imageInfo["height"]
    widthScaleFactor = newWidth / oldWidth
    heightScaleFactor = newHeight / oldHeight
    return (widthScaleFactor, heightScaleFactor)


def scaleAnnotations(
    oldWidth, oldXValue, widthScaleFactor, oldHeight, oldYValue, heightScaleFactor
):
    newWidth = oldWidth * widthScaleFactor
    newXValue = oldXValue * widthScaleFactor
    newHeight = oldHeight * heightScaleFactor
    newYValue = oldYValue * heightScaleFactor
    return (newWidth, newXValue, newHeight, newYValue)


def resizeBoundingBoxes(annotations, scaleFactor):
    for annotation in annotations:
        boundingBox = annotation["bbox"]
        scaledValues = scaleAnnotations(
            boundingBox[2],
            boundingBox[0],
            scaleFactor[0],
            boundingBox[3],
            boundingBox[1],
            scaleFactor[1],
        )
        boundingBox[0] = scaledValues[1]
        boundingBox[1] = scaledValues[3]
        boundingBox[2] = scaledValues[0]
        boundingBox[3] = scaledValues[2]
    return annotations


def loadImage(imagePath):
    image = cv2.imread(imagePath)
    if image is None:
        raise ValueError(f"Image not found at path: {imagePath}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (244, 244))
    return image


def loadAnnotations(COCO, imageId):
    annotationIds = COCO.getAnnIds(imgIds=[imageId])
    annotations = COCO.loadAnns(annotationIds)
    return annotations


def padAnnotations(annotations, maxAnnotations=10):
    paddedAnnotations = np.zeros((maxAnnotations, 4))
    for i, annotation in enumerate(annotations):
        if i >= maxAnnotations:
            break
        paddedAnnotations[i, :] = annotation["bbox"]
    return paddedAnnotations


def generateData(COCO, imageIds, animal, axAnnotations=10):
    run = 0
    for imageId in imageIds:
        run += 1
        imageInfo = COCO.loadImgs([imageId])[0]
        scaleFactor = calculateScaleFactor(imageInfo)
        imagePath = f"/content/drive/My Drive/Training Set/{animal}/Images/{imageInfo['file_name']}"
        image = loadImage(imagePath)
        annotation = loadAnnotations(COCO, imageId)
        annotation = resizeBoundingBoxes(annotation, scaleFactor)
        paddedAnnotation = padAnnotations(annotation)
        if run % 25 == 0:
            print(
                f"Loaded image {imageId} with shape {image.shape} and annotations {paddedAnnotation.shape}"
            )
    yield image, paddedAnnotation


model = tf.keras.applications.MobileNetV2(
    input_shape=(244, 244, 3), include_top=False, weights="imagenet"
)
model = tf.keras.Sequential(
    [
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
catTrainingDataset = tf.data.Dataset.from_generator(
    lambda: generateData(catCOCO, catImageIds, "Cats"),
    output_signature=(
        tf.TensorSpec(shape=(244, 244, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1), dtype=tf.float32),
    ),
)
dogTrainingDataset = tf.data.Dataset.from_generator(
    lambda: generateData(dogCOCO, dogImageIds, "Dogs"),
    output_signature=(
        tf.TensorSpec(shape=(244, 244, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1), dtype=tf.float32),
    ),
)
trainingDataset = catTrainingDataset.concatenate(dogTrainingDataset)
trainingDataset = (
    trainingDataset.shuffle(buffer_size=100).batch(32).prefetch(tf.data.AUTOTUNE)
)

trainedModel = model.fit(trainingDataset, epochs=10)
