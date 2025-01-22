import json
import tensorflow as tf
from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image

# Load the COCO dataset
catAnnotationFile = "Training/Training Data/Training Set/Cats/cats.json"
dogAnnotationFile = "Training/Training Data/Training Set/Dogs/dogs.json"
catCOCO = COCO(catAnnotationFile)
print("Loaded COCO dataset for cats")
dogCOCO = COCO(dogAnnotationFile)
print("Loaded COCO dataset for dogs")

catImageIds = list(catCOCO.imgs.keys())
print(f"Loaded {len(catImageIds)} images for cats")
dogImageIds = list(dogCOCO.imgs.keys())
print(f"Loaded {len(dogImageIds)} images for dogs")


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
    paddedAnnotations = np.zeros((maxAnnotations, 5))
    for i, annotation in enumerate(annotations):
        if i >= maxAnnotations:
            break
        boundingBox = annotation["bbox"]
        categoryId = annotation["category_id"]
        paddedAnnotations[i, :] = [
            boundingBox[0],
            boundingBox[1],
            boundingBox[2],
            boundingBox[3],
            categoryId,
        ]
    return paddedAnnotations


def generateData(COCO, imageIds, animal, axAnnotations=10):
    run = 0
    totalImages = len(imageIds)
    for imageId in imageIds:
        run += 1
        imageInfo = COCO.loadImgs([imageId])[0]
        scaleFactor = calculateScaleFactor(imageInfo)
        imagePath = f"Training/Training Data/Training Set/{animal}/Images/{imageInfo['file_name']}"
        image = loadImage(imagePath)
        annotation = loadAnnotations(COCO, imageId)
        annotation = resizeBoundingBoxes(annotation, scaleFactor)
        paddedAnnotation = padAnnotations(annotation)
        if run % 500 == 0:
            print(annotation)
            print(paddedAnnotation)
            print(
                f"Loaded image {imageId} of {totalImages} with shape {image.shape} and annotations {paddedAnnotation.shape}"
            )
    yield image, paddedAnnotation


model = tf.keras.applications.MobileNetV2(
    input_shape=(244, 244, 3), include_top=False, weights="imagenet"
)
model = tf.keras.Sequential(
    [
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(50),
        tf.keras.layers.Reshape((10, 5)),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
catTrainingDataset = tf.data.Dataset.from_generator(
    lambda: generateData(catCOCO, catImageIds, "Cats"),
    output_signature=(
        tf.TensorSpec(shape=(244, 244, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(10, 5), dtype=tf.float32),
    ),
)
dogTrainingDataset = tf.data.Dataset.from_generator(
    lambda: generateData(dogCOCO, dogImageIds, "Dogs"),
    output_signature=(
        tf.TensorSpec(shape=(244, 244, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(10, 5), dtype=tf.float32),
    ),
)
trainingDataset = catTrainingDataset.concatenate(dogTrainingDataset)
trainingDataset = (
    trainingDataset.shuffle(buffer_size=100).batch(32).prefetch(tf.data.AUTOTUNE)
)

trainedModel = model.fit(trainingDataset, epochs=10)

# Save the model
model.save("Model/catDogDetector.h5")
