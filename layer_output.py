from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
import cv2
import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt


labels={ 0 : "CAT",
         1 : "DOG",
         2 : "HORSE"}

def plot_feature_maps(feature_maps):
    """
    uncomment to see all maps
    :param feature_maps:
    :return:
    """
    # height, width, depth = feature_maps.shape
    # nb_plot = int(np.rint(np.sqrt(depth)))
    # fig = plt.figure(figsize=(30, 30))
    # for i in range(depth):
    #     plt.subplot(nb_plot, nb_plot, i+1)
    #     plt.imshow(feature_maps[:,:,i])
    #     plt.title('feature map {}'.format(i+1))
    # plt.show()
    plt.imshow(feature_maps[:, :, 20])
    plt.show()



# load model
model = load_model('model.h5')
model_layers = [ layer.name for layer in model.layers]
print(model_layers)
features_extractor = Model(inputs=model.input, outputs=model.get_layer(model_layers[0]).output)

orig = cv2.imread('test/17.jpg')
image = cv2.resize(orig, (124, 124))
image = image.astype("float") / 255.0
image = np.expand_dims(image, axis=0)
feature_maps = features_extractor.predict(image)[0]
plot_feature_maps(feature_maps)