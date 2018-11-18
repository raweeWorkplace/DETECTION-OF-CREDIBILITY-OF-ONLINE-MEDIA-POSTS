from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def tokenize(texts,labels,maxlen,max_words):
    tokenizer = Tokenizer(num_words=max_words)

    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens in the training dataset.' % len(word_index))   

    data = pad_sequences(sequences, maxlen=maxlen)
    print('-'*80 )
    print('Printing Length of dataset')
    print(len(data))
    print('-'*80)
    

    labels = np.asarray(labels)
    print('-'*80 )
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    print("Tokenizing performed successfully on dataset.")
    print('*'*80)
    
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    x_train = data
    y_train = labels

    return (x_train,y_train,word_index)