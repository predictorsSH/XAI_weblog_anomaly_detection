from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Reshape
from tensorflow.keras import Model, Sequential


class CNN():

    def __init__(self):
        self.input_shape = 121


    def build_model(self):

        model = Sequential()
        model.add(Reshape((121,1), input_shape=(self.input_shape,1)))
        model.add(Conv1D(32,4))
        # model.add(Conv1D(32,4, input_shape=(self.input_shape,1)))
        model.add(MaxPool1D(2))
        model.add(Conv1D(32, 4))
        model.add(MaxPool1D(2))
        model.add(Conv1D(16, 4))
        model.add(MaxPool1D(2))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        print(model.summary())

        return model

    def learn(self, model, train, target):
        model.fit(train, target, batch_size=100 ,epochs=1,validation_split=0.2)




