import os
import numpy as np
glove_dir = 'glove.6B/'
embeddings_index = {}
print(os.path.join(glove_dir,'glove.6B.100d.txt'))
f = open(os.path.join(glove_dir,'glove.6B.100d.txt'))

for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 100
def embedd(max_words,word_index,embedding_matrix):
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if i < max_words:
	        if embedding_vector is not None:
	            # Words not found in embedding index will be all-zeros.
	            embedding_matrix[i] = embedding_vector
	

def generate_matrix(max_words,word_index):
	embedding_matrix = np.zeros((max_words, embedding_dim))
	embedd(max_words,word_index,embedding_matrix)
	return(embedding_matrix,embedding_dim)