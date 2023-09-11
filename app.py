import os

from sklearn.cluster import KMeans
from PIL import Image
from collections import Counter
import numpy as np
import cv2


def get_image(pil_image):
    nimg = np.array(pil_image)
    image = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_labels(rimg):
    clf = KMeans(n_clusters=5, n_init=10)
    labels = clf.fit_predict(rimg)
    return labels, clf


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_colours(pimg):
    img = get_image(pimg)
    reshaped_img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    labels, clf = get_labels(reshaped_img)
    counts = Counter(labels)
    center_colours = clf.cluster_centers_
    ordered_colours = [center_colours[i] for i in counts.keys()]
    hex_colours = [RGB2HEX(ordered_colours[i]) for i in counts.keys()]
    return hex_colours


images = os.listdir("images")

for image in images:
    pil_image = Image.open("images/" + image)
    hex_colours = get_colours(pil_image)
    print(hex_colours)
