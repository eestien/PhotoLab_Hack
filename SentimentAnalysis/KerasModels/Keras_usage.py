import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D, GRU

def Keras_usage(data):
    maxlen = 100
    with open('../model/tokenizer_Keras.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    vocab_size = len(loaded_tokenizer.word_index) + 1
    model = tf.keras.models.load_model('../model/Keras_model.h5')


    txt = data
    seq = loaded_tokenizer.texts_to_sequences([txt])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict_classes(padded)
    print(pred)

print(tf.__version__)
Keras_usage('i hate you')