from sklearn.feature_extraction import text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# def tfidfv2(train, test):
#     counter = text.CountVectorizer(lowercase=False)
#     counter.

class Tfidf_array():

    def __init__(self, max_features=None):
        self.tovec=text.TfidfVectorizer(lowercase=False, max_features=max_features)


    def fit(self,train):
        train_vectors = self.tovec.fit_transform(train)
        train_vectors = train_vectors.toarray()

        return self.tovec, train_vectors

    def transform(self,test):

        test_vector = self.tovec.transform(test)
        test_vector = test_vector.toarray()

        return test_vector


def tfidf(train, test):

    tovec = text.TfidfVectorizer(lowercase=False,use_idf=True)
    train_vectors = tovec.fit_transform(train)
    test_vectors = tovec.transform(test)


    return tovec, train_vectors, test_vectors

def label_encoding(train,test):

    tovec = Tokenizer(filters=' ')
    tovec.fit_on_texts(list(train))

    seq_train = tovec.texts_to_sequences(train)
    pad_train = pad_sequences(seq_train)
    N = pad_train.shape[1]

    seq_test = tovec.texts_to_sequences(test)
    pad_test = pad_sequences(seq_test,maxlen=N)

    return tovec, pad_train, pad_test


class Text2vec():

    def __init__(self):

        self.tovec = Tokenizer(filters=' ')

    def fit(self,train):

        self.tovec.fit_on_texts(list(train))
        vec= self.tovec.texts_to_sequences(train)
        vec = pad_sequences(vec, maxlen=512)
        vec = vec.reshape(-1,512,1)

        return vec


    def transform(self, text):

        vec = self.tovec.texts_to_sequences([text])
        print(vec)
        vec = pad_sequences(vec, maxlen=512)
        vec = vec.reshape(-1,512,1)
        print(vec.shape)

        return vec






