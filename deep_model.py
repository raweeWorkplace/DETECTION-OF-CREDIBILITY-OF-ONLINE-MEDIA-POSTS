from keras.layers import LSTM, Dense, Embedding, Flatten, Dropout
from keras.models import Sequential

import csv_check as cs
import embedding_layer as emb_layer
import load_data as ld
import text_tokenizing as tt
from sklearn.model_selection import train_test_split

#Some constant values
maxlen = 250  # We will cut reviews after 100 words
max_words = 50000  # We will only consider the top 50,000 words in the dataset


#labeling the data set
texts,labels = cs.load_csv()
print(len(texts))
print(texts.head())

#spliting into train and test
X_train, X_test, y_train, y_test = train_test_split(texts, labels,test_size=0.2)
print(X_train.head())

#sampling the text data
x_train,y_train,word_index = tt.tokenize(X_train,y_train,maxlen,max_words)
X_test, y_test, words= tt.tokenize(X_test,y_test,maxlen,max_words)

embedding_matrix, embedding_dim = emb_layer.generate_matrix(max_words,word_index)

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
#model.add(Flatten())
model.add(LSTM(maxlen))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2, input_shape=(100,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=50,
                    #batch_size=500,
                    validation_data=(X_test, y_test))

model_json = model.to_json()
with open("model21.json", "w") as json_file:
	json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights('pre_trained_glove_model.h5')
	print("Saved model to disk")
#plotting the results

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()