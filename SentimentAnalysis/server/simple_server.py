from flask import Flask, request
from joblib import dump, load
import matplotlib.pyplot as plt
import cv2
import io
import base64
import pychubby
from pychubby.actions import Chubbify, Multiple, Pipeline, Smile
from pychubby.detect import LandmarkFace
app = Flask(__name__)



from nltk.corpus import stopwords
from textblob import Word
import joblib
import numpy as np


def analyze(data):
    answers_decode = {0: [
        "#616161",
        "#9E9E9E",
        "#757575",
        "#536DFE",
        "#607D8B"
        ], 1: [
        "#00BCD4",
        "#B3E5FC",
        "#03A9F4",
        "#00BCD4",
        "#00796B"
        ], 2: [
        "#00796B",
        "#B2DFDB",
        "#009688",
        "#757575",
        "#FFC107"
        ], 3:  [
        "#7B1FA2",
        "#E1BEE7",
        "#FF4081",
        "#9C27B0",
        "#D32F2F"
        ], 4: [
        "#E64A19",
        "#D32F2F",
        "#FFEB3B",
        "#E040FB",
        "#F44336"
        ], 5: [
        "#FFA000",
        "#FFECB3",
        "#FFC107",
        "#CDDC39",
        "#FFA000"
        ]
    }
    data = data.replace('[^\w\s].',' ').split()
    stop = stopwords.words('english')
    data = list(map(lambda x: " ".join(x for x in x.split() if x not in stop), data))
    data = list(map(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]), data))
    count_vect = joblib.load('../model/class_triple.joblib')
    #count_vect = joblib.load(cf.EMBEDDINGS_PATH)
    data_vect = count_vect.transform(data)
    rf = joblib.load('../model/rf_triple.joblib')
    #rf = joblib.load(cf.MODEL_PATH)
    data_pred = list(rf.predict(data_vect))
    data_pred = max(set(data_pred), key=data_pred.count)
    answ = answers_decode.get(data_pred)
    return answ






@app.route('/')
def main():
    return 'Hi :)'

@app.route('/photo')
def photo():
    img = request.args.get('photob62')
    imgdata = base64.b64decode(img)
    img = cv2.imread(io.BytesIO(base64.b64decode(imgdata)))
    img = cv2.imread(img)
    img8 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lf = LandmarkFace.estimate(img8)

    from pychubby.actions import Smile, OpenEyes, Multiple, RaiseEyebrow, StretchNostrils, AbsoluteMove

    smile = Smile(scale=0.2)


    new_lf, df = smile.perform(lf)  # lf defined above

    # new_lf.plot(show_landmarks=False)
    plt.imsave('output_image.png', new_lf.img)
    import base64
    encoded = base64.b64encode(open("output_image.png", "rb").read())
    return encoded


@app.route('/text')
def text():
    message = request.args.get('message')
    emo_color = analyze(message)


    return {'colors': emo_color}
    # return 'Price {0} \n Estimated Sale Time: {1} days'.format(price, term)

if __name__ == '__main__':
    app.run(debug=True, port='5000', host='localhost')
