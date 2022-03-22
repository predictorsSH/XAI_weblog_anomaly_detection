import pandas as pd
from preprocessing import vectorizer
from models import CNN
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

if __name__ == '__main__':
    #data load
    data=pd.read_csv('../data/spam_ham_dataset.csv')

    #tfidf
    tovec = vectorizer.Tfidf_array(max_features=121)
    #train data vectorize
    tfidf_tovec, train_vec = tovec.fit(data['text'])
    #test data vectorize, In this example, two data sets are identical.
    test_vec = tovec.transform(data['text'])


    #target variable to categorical
    target = to_categorical(data['label_num'])


    #train
    model=CNN.CNN(feature_size=train_vec.shape[1],batch_size=10, epochs=10)
    clf = model.build_model()
    model.learn(clf, train_vec, target)

    #predict
    pred = clf.predict(test_vec)
    prediction = np.argmax(pred,axis=1)
    print(classification_report(data['label_num'], prediction))

    #pipeline
    pipe = make_pipeline(tovec, clf)

    #LIME 설명체 선언
    class_names=['0','1']
    explainer = LimeTextExplainer(random_state=42)

    #설명
    exp = explainer.explain_instance(data['text'][3], pipe.predict, top_labels=1)
    exp.save_to_file('../output/lime/spam_cnn.html')


