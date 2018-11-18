from keras.layers import LSTM, Dense, Embedding, Flatten, Dropout
from keras.models import Sequential

import csv_check as cs
import embedding_layer as emb_layer
import load_data as ld
import text_tokenizing as tt
from sklearn.model_selection import train_test_split

#Some constant values
maxlen = 500  # We will cut reviews after 100 words
max_words = 50000 # We will only consider the top 50,000 words in the dataset


#labeling the data set
texts,labels = cs.load_csv()
print(labels)

#spliting into train and test
X_train, X_test, y_train, y_test = train_test_split(texts, labels,test_size=0.2)

#sampling the text data
print('\n'+'*'*80 )
print('Tokenizer Fittig on Training dataset....')
print('-'*80)
x_train,y_train,word_index = tt.tokenize(X_train,y_train,maxlen,max_words)
print('\n'+'*'*80 )
print('Tokenizer Fittig on Test dataset........')
print('-'*80)
X_test, y_test, words= tt.tokenize(X_test,y_test,maxlen,max_words)

embedding_matrix, embedding_dim = emb_layer.generate_matrix(max_words,word_index)

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(LSTM(maxlen,return_sequences=True))
model.add(LSTM(maxlen))
model.add(Dropout(0.5))
model.add(Dense(2500, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print('\n'+'*'*80 )
print('Printing the proposed model summary........')
print('-'*80)
model.summary()


model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
print('\n'+'*'*80 )
print('Compiling the proposed model summary........')
print('-'*80 )
print('Printing the proposed model JSON FILE........')
print('-'*80)
model_json = model.to_json()
print(model_json)
print('-'*80)

history = model.fit(x_train, y_train,
                    epochs=6,
                    batch_size=100,
                    validation_data=(X_test, y_test))

print('\n'+'*'*80 )
print('Training the proposed model ........')
print('-'*80 )
print('Saving Trained Model to file : Pre_Trained_Global_Model........')
print('-'*80)
with open("model21.json", "w") as json_file:
	json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights('pre_trained_glove_model.h5')
	print("Saved model to disk")


# #plotting the results
print('\n'+'*'*80 )
print('Plotting the results ........')
print('-'*80 )

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Testing acc')
plt.title('Training and Text accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training and Test loss')
plt.legend()
plt.show()