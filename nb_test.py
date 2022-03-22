from preprocessing import data_load, vectorizer
from sklearn.naive_bayes import MultinomialNB
import random
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import tensorflow as tf
import time
from sklearn.metrics import classification_report
import pandas as pd

tf.config.experimental_run_functions_eagerly(True)
gpu = tf.config.experimental.list_physical_devices('GPU')


#model
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


    #vectorize
    tovec, train_vec, test_vec = vectorizer.tfidf(train_x, test_x)


    #nb train, fitting
    nb.fit(train_vec,train_target)

    # nb test, predict
    pred = nb.predict(test_vec)
    pred[:10]
    # f1_score(test_target, pred, pos_label='1', average='weighted')

    pred_target=pd.DataFrame(list(zip(pred, test_target)), columns=['pred','target'])

    for i in range(len(pred_target)):
        if (pred_target.iloc[i]['pred'] == 0) & (pred_target.iloc[i]['target'] == 0) :
            print(pred_target.iloc[i],i)

    print(test_target[:10])

    print(classification_report(test_target , pred))

    #pipeline
    pipe = make_pipeline(tovec, nb)
    predict_classes = pipe.predict_proba([test_x[0]])


    #LIME 설명체 선언
    class_names=[0,1]
    explainer = LimeTextExplainer(class_names=class_names)

    pred[:20]
    test_target[:20]
    #Lime 설명 및 저장

    start = time.time()

    # for i in range(10):
    #     file_name = 'lime' + str(i) +'.html'
    #     exp = explainer.explain_instance(test_x[i], pipe.predict_proba ,labels=[1],top_labels=1)
    #     exp.save_to_file('output/lime/'+file_name)
    #
    # print('time:', time.time() - start)

    exp = explainer.explain_instance(test_x[9818], pipe.predict_proba ,labels=[1],top_labels=1)
    exp.save_to_file('output/lime/9818.html')