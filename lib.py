
# coding: utf-8

# In[35]:

import keras
import numpy
import pickle



from keras.preprocessing.image import list_pictures, load_img, img_to_array




import os
import random
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Reshape, BatchNormalization




def load_and_resize_img(k):
    size = (64, 64)
    img = load_img(os.path.join(os.pardir, "train", k)).resize(size)
    return img


def prepare_dataset(img_names):
    n = len(img_names)
    resized = [load_and_resize_img(c) for c in img_names]
    arrays = [img_to_array(x) for x in resized]
    array = numpy.array(arrays)
    assert array.shape == (n, 64, 64, 3)
    return array


def prepare_train_test():
    pictures = os.listdir(os.path.join(os.pardir, "train"))
    cats, dogs = [p for p in pictures if p.startswith('cat')], [p for p in pictures if p.startswith('dog')]
    num = 4000
    random.shuffle(cats)
    cats_t_paths = cats[:num]
    random.shuffle(dogs)
    dogs_t_paths = dogs[:num]


    print("Prepare train")
    train_data_X = numpy.concatenate([prepare_dataset(cats[0:num]), prepare_dataset(dogs[0:num])])
    train_data_y = numpy.array([0] * num + [1] * num)

    print("Prepare test")
    test_data_X = numpy.concatenate([prepare_dataset(cats[num:2 * num]), prepare_dataset(dogs[num:2 * num])])
    test_data_y = numpy.array([0] * num + [1] * num)

    return train_data_X, train_data_y, test_data_X, test_data_y


def prepare_for_logreg(dataset):
    array = numpy.array([a.reshape(-1) for a in dataset])
    assert array.shape[1:] == (64 * 64 * 3,), array.shape
    return array


def train_logreg():
    # In[ ]:
    print("Train logreg")
    from sklearn.linear_model import LogisticRegression
    logreg_m = LogisticRegression()
    logreg_m.fit(prepare_for_logreg(train_data_X), train_data_y,
         validation_data=(test_data_X, test_data_y))



from sklearn.metrics import accuracy_score

def score(model, X, expected_y, process_predicted=lambda x: x):
    predicted = model.predict(X)
    return accuracy_score(expected_y, process_predicted(predicted))
    
def score_both(model, training_X, training_y, test_X, test_y, process_predicted=lambda x: x):
    print("Train accuracy", score(model, training_X, training_y, process_predicted))
    print("Test accuracy", score(model, test_X, test_y, process_predicted))


# from https://keras.io/layers/convolutional/


def get_convnet_model(input_shape):
    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))

    nb_filters = 32
    nb_pool = 2
    nb_conv = 3

    model.add(Dropout(0.2))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.5))


    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, init='zero', activation='sigmoid'))

    from keras.optimizers import SGD
    sgd = SGD(lr=1)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model



def interactive_train_convnet(model, d):
    train_data_X, train_data_y, test_data_X, test_data_y = d

    print("train convnet")
    print(len(train_data_X), train_data_y[0], train_data_y[-1])
    model.fit(train_data_X, train_data_y, nb_epoch=60)


    # In[ ]:
    print("predict from convnet")
    post_predict = lambda x: numpy.round(x.reshape(-1)).astype(int)
    print(post_predict(model.predict(train_data_X)))
    print(train_data_y)


    # In[ ]:
    print("score convnet")
    score_both(model, train_data_X, train_data_y, test_data_X, test_data_y, post_predict)


def save_data(data):
    with open("data.pickle", "wb") as f:
        pickle.dump(data, f)


def load_data():
    with open("data.pickle", "rb") as f:
        return pickle.load(f)


