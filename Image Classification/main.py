import cv2 as cv
import numpy as np
import matplotlib.pyplot as pit
from tensorflow.keras import datasets,layers,models
import os

print("Current working directory:", os.getcwd())
print("Image exists:", os.path.exists("plane.jpg"))

(training_images,training_labels),(test_images,test_labels)=datasets.cifar10.load_data()
training_images, test_images = training_images / 255.0, test_images / 255.0
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

for i in range(16):
    pit.subplot(4,4,i+1)
    pit.xticks([])
    pit.yticks([])
    pit.imshow(training_images[i],cmap=pit.cm.binary)
    pit.xlabel(class_names[training_labels[i][0]])

pit.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
test_images = test_images[:4000]    
test_labels = test_labels[:4000]


model = models.load_model("image_classification_model.keras",compile=False)
img = cv.imread('plane.jpg')
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
img = cv.resize(img,(32,32))

pit.imshow(img,cmap=pit.cm.binary)

prediction = model.predict(np.array([img])/255.0)
predicted_class = class_names[np.argmax(prediction)]
print(f'Predicted class: {predicted_class}')
pit.xlabel(predicted_class)

