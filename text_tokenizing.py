from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

training_samples = 1000 # We will be training on 60000 samples
validation_samples = 280 # We will be validating on 20000 samples

def tokenize(texts,labels,maxlen,max_words):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=maxlen)
    print(len(data))

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # Split the data into a training set and a validation set
    # But first, shuffle the data, since we started from data
    # where sample are ordered (all negative first, then all positive).
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # x_train = data[:training_samples]
    # y_train = labels[:training_samples]
    x_train = data
    y_train = labels
    # x_val = data[training_samples: training_samples + validation_samples]
    # y_val = labels[training_samples: training_samples + validation_samples]

    return (x_train,y_train,word_index)