from pychubby.visualization import create_animation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import cv2
import ffmpeg
from matplotlib import animation
from matplotlib.animation import Animation, PillowWriter
import pychubby
from pychubby.actions import Chubbify, Multiple, Pipeline, Smile
from pychubby.detect import LandmarkFace
import numpy as np
def smile(img):
    img_path = img
    img = cv2.imread(img_path)
    img8 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lf = LandmarkFace.estimate(img8)
    from pychubby.actions import Smile

    a = Smile(scale=0.2)
    new_lf, df = a.perform(lf)
    ani = create_animation(df, img)
    plt.imsave('output_image.gif', ani)

    plt.show()

