import matplotlib.pyplot as plt
import cv2
import pychubby
from pychubby.actions import Chubbify, Multiple, Pipeline, Smile
from pychubby.detect import LandmarkFace
import numpy as np
def smile(img):
    img_path = img
    img = cv2.imread(img_path)
    img8 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lf = LandmarkFace.estimate(img8)

    from pychubby.actions import Action, Smile, OpenEyes, Multiple, RaiseEyebrow, StretchNostrils, AbsoluteMove

    smile = OpenEyes(scale=0.1)


    new_lf, df = smile.perform(lf)  # lf defined above

    # new_lf.plot(show_landmarks=False)
    plt.imsave('output_image_man1.png', new_lf.img)
    #import base64
    #encoded = base64.b64encode(open("output_image.png", "rb").read())
    #return encoded


