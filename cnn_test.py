from preprocessing import data_load, vectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer
from models import CNN
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import classification_report
import time


tf.config.experimental_run_functions_eagerly(True)
gpu = tf.config.experimental.list_physical_devices('GPU')

import matplotlib.pyplot as plt

nb = MultinomialNB(alpha=0.01)



if __name__ == '__main__':

    #data load
    train_path = 'data/si/SI_train.done'
    test_path = 'data/si/SI_val.done'

    train = data_load.json_load(train_path)
    test = data_load.json_load(test_path)

    #data suffle
    random.shuffle(train)
    random.shuffle(test)


    #paylaod, target 분리

    train_x = []
    train_target = []
    test_x = []
    test_target = []

    train_x[:10]

    for d in train:
        train_x.append(d['payload'])
        train_target.append(d['label'])


    for d in test:
        test_x.append(d['payload'])
        test_target.append(d['label'])

    for i in range(len(train_target)):
        if train_target[i] == '0':
            train_target[i] = 0
        elif train_target[i] == '1':
            train_target[i] = 1

    for i in range(len(test_target)):
        if test_target[i] == '0':
            test_target[i] = 0
        elif test_target[i] == '1':
            test_target[i] = 1



    # tokenizer, pad_train, pad_test = vectorizer.label_encoding(list(train_x), test_x)
    # pad_train.shape
    # pad_train=pad_train.reshape(-1, 6688,1)
    # pad_test=pad_test.reshape(-1,6688,1)
    new_train_target = np.array(train_target).reshape(-1,1)
    new_train_target=to_categorical(new_train_target)
    new_test_target = np.array(test_target).reshape(-1,1)
    new_test_target = to_categorical(new_test_target)

    #tfidf
    tovec = vectorizer.Tfidf_array()
    tfidftovec, train_vec = tovec.fit(train_x)
    test_vec = tovec.transform(test_x)
    len(tfidftovec.get_feature_names())

    #모델 학습
    model=CNN.CNN()
    clf = model.build_model()
    model.learn(clf, train_vec, new_train_target)

    #predict
    pred = clf.predict(test_vec)
    prediction = np.argmax(pred,axis=1)
    print(classification_report(test_target , prediction))

    #pipeline
    pipe = make_pipeline(tovec, clf)
    predict_classes = pipe.predict_proba([test_x[0]])


    #LIME 설명체 선언
    class_names=['0','1']
    explainer = LimeTextExplainer(random_state=42)


    exp = explainer.explain_instance(test_x[21], pipe.predict_proba, top_labels=1)
    exp.save_to_file('output/lime/cnn21.html')

    start = time.time()

    for i in range(10):
        file_name = 'CNNlime' + str(i) +'.html'
        exp = explainer.explain_instance(test_x[i], pipe.predict_proba , top_labels=1)
        exp.save_to_file('output/lime/'+file_name)

    print("time: ", time.time() - start)

    print(tf.__version__)