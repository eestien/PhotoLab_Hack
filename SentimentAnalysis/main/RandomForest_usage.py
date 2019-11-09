from nltk.corpus import stopwords
from textblob import Word
import joblib
import numpy as np


def analyze(data):
    data = data.replace('[^\w\s].',' ').split()
    stop = stopwords.words('english')
    data = list(map(lambda x: " ".join(x for x in x.split() if x not in stop), data))
    data = list(map(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]), data))
    count_vect = joblib.load('../model/class_rf.joblib')
    data_vect = count_vect.transform(data)
    rf = joblib.load('../model/rf.joblib')
    data_pred = list(rf.predict(data_vect))
    data_pred = max(set(data_pred), key=data_pred.count)
    print(data_pred)

'''

tweetss = pd.DataFrame(['I am very happy today! The atmosphere looks cheerful',
'Things are looking great. It was such a good day',
'Success is right around the corner. Lets celebrate this victory',
'Everything is more beautiful when you experience them with a smile!',
'Now this is my worst, okay? But I am gonna get better.',
'I am tired, boss. Tired of being on the road, lonely as a sparrow in the rain. I am tired of all the pain I feel',
'This is quite depressing. I am filled with sorrow',
'His death broke my heart. It was a sad day'])
'''
'''
def analyze(tweets):
    # Doing some preprocessing on these tweets as done before
    tweets[0] = tweets[0].str.replace('[^\w\s]',' ')
    from nltk.corpus import stopwords
    stopp = stopwords.words('english')
    tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stopp))
    from textblob import Word
    tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # Extracting Count Vectors feature from our tweets
    count_vect = joblib.load('../model/class_rf.joblib')
    tweet_count = count_vect.transform(tweets[0])

    rf = joblib.load('../model/rf.joblib')
    #Predicting the emotion of the tweet using our already trained linear SVM
    tweet_pred = rf.predict(tweet_count)
    print(tweet_pred)
'''
analyze('Oh, shit i am lost all')