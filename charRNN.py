from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

import sys
import random
import numpy as np

data = 'data/input.txt'
entireText = open(data).read().lower()

uniqueChars = sorted(list(set(entireText)))
lenUniqueChars = len(uniqueChars)

modelParams = {
    'epochs' : 15,
    'maxlen' : 140,
    'step' : 3,
    'batch_size' : 256
}

def buildComputationGraph():
    model = Sequential()
    model.add(LSTM(128, input_shape=(modelParams['maxlen'], len(uniqueChars))))
    model.add(Dense(len(uniqueChars)))
    model.add(Activation('softmax'))

    # optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='Adagrad')
    return model


def breakText():
    sentences = []
    next_chars = []
    for i in range(0, len(entireText) - modelParams['maxlen'], modelParams['step']):
        sentences.append(entireText[i: i + modelParams['maxlen']])
        next_chars.append(entireText[i + modelParams['maxlen']])
    return [sentences, next_chars]

def mapChars():
    chartoIdx = {}
    idxtoChar = {}
    for idx, c in enumerate(uniqueChars):
        chartoIdx[c] = idx
        idxtoChar[idx] = c
    return [chartoIdx, idxtoChar]

# embed using One-Hot Encoding
def embedText(sentences, next_chars, chartoIdx):
    # X is the encoding of the sentences
    # shape of each sentence is (modelParams['maxlen'], len(uniqueChars))
    X = np.zeros((len(sentences), modelParams['maxlen'], len(uniqueChars)), dtype=np.bool)

    # Y is the encoding of the next characters to predict
    # shape of each next character is (len(uniqueChars))
    Y = np.zeros((len(sentences), len(uniqueChars)), dtype=np.bool)

    # embed text
    for i, sent in enumerate(sentences):
        for idx, char in enumerate(sent):
            X[i, idx, chartoIdx[char]] = 1
        Y[i, chartoIdx[next_chars[i]]] = 1

    return [X, Y]

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def runSession(model, X, y, idxtoChar):
    for iteration in range(1, 2):
        print('*' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size = modelParams['batch_size'], nb_epoch = modelParams['epochs'])

        start_index = random.randint(0, len(entireText) - modelParams['maxlen'] - 1)

        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- temperature:', temperature)

            generated = ''
            sentence = entireText[start_index: start_index + modelParams['maxlen']]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            print(generated)

            for i in range(400):
                x = np.zeros((1, modelParams['maxlen'], len(uniqueChars)))
                for t, char in enumerate(sentence):
                    x[0, t, chartoIdx[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = idxtoChar[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

[sentences, next_chars] = breakText()
[chartoIdx, idxtoChar] = mapChars()
[X, Y] = embedText(sentences, next_chars, chartoIdx)
model = buildComputationGraph()
runSession(model, X, Y, idxtoChar)
