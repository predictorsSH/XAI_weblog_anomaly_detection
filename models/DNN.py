from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Sequential


class CNN():

    def __init__(self):
        self.input_shape = 121


    def build_model(self):

        model = Sequential()
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        print(model.summary())

        return model

    def learn(self, model, train, target):
        model.fit(train, target, batch_size=100 ,epochs=1,validation_split=0.2)

