from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import os

labels={ 0 : "CAT",
         1 : "DOG",
         2 : "HORSE"}

if __name__ =="__main__":

    # load model
    model = load_model('model.h5')
    # summarize model.
    # model.summary()


    for x in os.listdir("test"):

        orig = cv2.imread('test/'+x)
        image = cv2.resize(orig, (124, 124))
        image = image.astype("float") / 255.0
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        label_id = np.argmax(pred)
        label = labels.get(label_id)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (50, 50)
        fontScale = 1
        fontColor = (0, 0, 0)
        lineType = 2

        cv2.putText(orig, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow("Prediction",orig)
        print(label)
        cv2.waitKey(1000)