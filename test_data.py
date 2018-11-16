import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
imdb_dir = 'text/ProjectDemo/IMDB'
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []

maxlen = 100  # We will cut reviews after 50 words
max_words = 10000 
def load_test_data():
	for label_type in ['neg', 'pos']:
		dir_name = os.path.join(test_dir, label_type)
		for fname in sorted(os.listdir(dir_name)):
			if fname[-4:] == '.txt':
				f = open(os.path.join(dir_name, fname))
				texts.append(f.read())
				f.close()
				if label_type == 'neg':
					labels.append(0)
				else:
					labels.append(1)
	tokenizer = Tokenizer(num_words=max_words)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)
	x_test = pad_sequences(sequences, maxlen=maxlen)
	y_test = np.asarray(labels)
	return(x_test,y_test)