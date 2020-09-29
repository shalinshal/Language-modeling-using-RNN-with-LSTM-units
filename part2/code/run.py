# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:19:30 2019

@author: Shalin
"""

#from __future__ import print_function
import collections
import os
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
#from keras.optimizers import Adam
from keras.utils import to_categorical
#from keras.callbacks import ModelCheckpoint
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

DATA_PATH = "..\simple-examples\data"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_opt', type=str, default='train',choices=['train','test'], help='An integer: 1 to train, 2 to test')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='The full path of the training data')
    parser.add_argument('--char_word', type=str, default='', help='Train/Test character or word')
    return parser.parse_args()


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id
#    return dict(zip(counter.keys(),range(len(counter))))


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data(char_word=''):
    # get the data paths
    if char_word:
        char_word += '.'
    train_path = os.path.join(DATA_PATH, "ptb."+char_word+"train.txt")
    valid_path = os.path.join(DATA_PATH, "ptb."+char_word+"valid.txt")
    test_path = os.path.join(DATA_PATH, "ptb."+char_word+"test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

#    print(train_data[:5])
#    print(word_to_id)
#    print(vocabulary)
#    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

def perplexity(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    return K.pow(2.0, cross_entropy)


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y

def create_model():
#    use_dropout=True
    model = Sequential()
    model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    #model.add(LSTM(hidden_size, return_sequences=True))
#    if use_dropout:
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation('softmax'))
    
#    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy',perplexity])
    
    print(model.summary())
    return model
#checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

if __name__ == '__main__':
    
    num_steps = 30
    batch_size = 100
    hidden_size = 500
    num_epochs = 20
    args = get_args()
#    if args.char_word=='char':
#        train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data('char')
#    else:
#        train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()
    train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data(args.char_word)
    if args.data_path:
        DATA_PATH = args.data_path
    
    if args.run_opt == 'train':
        train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                                   skip_step=num_steps)
        valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                                   skip_step=num_steps)
        model = create_model()
        history = model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                            validation_data=valid_data_generator.generate(),
                            validation_steps=len(valid_data)//(batch_size*num_steps),verbose=2)#, callbacks=[checkpointer])
# =============================================================================
#         for key,values in iter(history.history):
#             history.history
#         loss_history = history.history['val_loss']#, 'val_categorical_accuracy', 'val_perplexity', 'loss', 'categorical_accuracy', 'perplexity']
#         numpy_loss_history = np.array(loss_history)
#         np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
# =============================================================================
        df = pd.DataFrame.from_dict(history.history)
        df.to_csv('word.csv')
        if args.char_word=='':
            word ='word'
        else:
            word = 'char'
        model.save(DATA_PATH +'\\' + word +"_model.hdf5")
        
        fig, ax = plt.subplots()
        ax.plot(range(num_epochs), history.history['perplexity'], '-b', label='training perplexity')
        ax.plot(range(num_epochs), history.history['val_perplexity'], '-r', label='validation perplexity')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perplexity')
        ax.legend()
        if args.char_word:
            ax.set_title('Character-level')
        else:
            ax.set_title('Word-level')
        plt.savefig('word-perplex.png')
        fig, ax = plt.subplots()
        ax.plot(range(num_epochs), history.history['categorical_accuracy'], '-b', label='training accuracy')
        ax.plot(range(num_epochs), history.history['val_categorical_accuracy'], '-r', label='validation accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        if args.char_word:
            ax.set_title('Character-level')
        else:
            ax.set_title('Word-level')
        plt.savefig('word-acc.png')
        
#    elif args.run_opt == 'test':
    else:
        if args.char_word=='':
            word ='word'
        else:
            word = 'char'
        model = load_model(DATA_PATH + "\\"+word+"_model.hdf5", custom_objects={'perplexity': perplexity})
        dummy_iters = 40
        example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, vocabulary,
                                                         skip_step=1)
        print("\nTraining data:")
        for i in range(dummy_iters):
            dummy = next(example_training_generator.generate())
        num_predict = 100
        true_print_out = "Actual words: \n"
        pred_print_out = "Predicted words: \n"
        for i in range(num_predict):
            data = next(example_training_generator.generate())
            prediction = model.predict(data[0])
            predict_word = np.argmax(prediction[:, num_steps-1, :])
            true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
            pred_print_out += reversed_dictionary[predict_word] + " "
        print(true_print_out,'\n')
        print(pred_print_out)
        # test data set
        dummy_iters = 40
        example_test_generator = KerasBatchGenerator(test_data, num_steps, 1, vocabulary,
                                                         skip_step=1)
        print("\nTest data:")
        for i in range(dummy_iters):
            dummy = next(example_test_generator.generate())
        num_predict = 200
        true_print_out = "Actual words: \n"
        pred_print_out = "Predicted words: \n"
        for i in range(num_predict):
            data = next(example_test_generator.generate())
            prediction = model.predict(data[0])
            predict_word = np.argmax(prediction[:, num_steps - 1, :])
            true_print_out += reversed_dictionary[test_data[num_steps + dummy_iters + i]] + " "
            pred_print_out += reversed_dictionary[predict_word] + " "
        print(true_print_out,'\n')
        print(pred_print_out)