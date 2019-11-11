import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D, GRU
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import keras
import tensorflow as tf




data = pd.read_csv("../data/text_emotion.csv")
data = data.drop('author', axis=1)

# Dropping rows with other emotion labels
data = data.drop(data[data.sentiment == 'anger'].index)
data = data.drop(data[data.sentiment == 'boredom'].index)
data = data.drop(data[data.sentiment == 'enthusiasm'].index)
data = data.drop(data[data.sentiment == 'empty'].index)
#data = data.drop(data[data.sentiment == 'sadness'].index)
data = data.drop(data[data.sentiment == 'fun'].index)
data = data.drop(data[data.sentiment == 'relief'].index)
data = data.drop(data[data.sentiment == 'surprise'].index)
#data = data.drop(data[data.sentiment == 'love'].index)
#data = data.drop(data[data.sentiment == 'hate'].index)
#data = data.drop(data[data.sentiment == 'neutral'].index)
#data = data.drop(data[data.sentiment == 'worry'].index)
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

X = []
sentences = list(data['content'])
for sen in sentences:
    X.append(preprocess_text(sen))

uniques = data.sentiment.unique()
for n, uni in enumerate(uniques):
    data.loc[data['sentiment'] == uni , 'sentiment'] = n
y = data.sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



print('BUild model ...')

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=maxlen))
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')


model.fit(X_train, y_train, batch_size=128, epochs=25, validation_data=(X_test, y_test), verbose=5)
# Save model
model.save('../model/Keras_model.h5')
loss, acc = model.evaluate(X_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))



# Save tokenizer
import pickle
with open('../model/tokenizer_Keras.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
