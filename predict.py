from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np

labels={ 0 : "CAT",
         1 : "DOG",
         2 : "HORSE"}
# load model
model = load_model('model.h5')
# summarize model.
model.summary()

orig = cv2.imread('test/h.jpg')


image = cv2.resize(orig, (124, 124))
image = image.astype("float") / 255.0
image = np.expand_dims(image, axis=0)
pred = model.predict(image)
label_id = np.argmax(pred)
label = labels.get(label_id)

cv2.imshow("Prediction",orig)
print(label)
cv2.waitKey()