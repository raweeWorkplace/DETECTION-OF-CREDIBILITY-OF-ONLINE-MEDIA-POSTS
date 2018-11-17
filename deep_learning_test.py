from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import tensorflow as tf
import test_data as td
# load json and create model
def getTest(x_itest,y_itest):
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("pre_trained_glove_model.h5")
	print("Loaded model from disk")

	model.compile(optimizer='rmsprop',
	              loss='binary_crossentropy',
	              metrics=['acc'])
	x_test,y_test = x_itest,y_itest
	clf_probs = model.evaluate(x_test,y_test)
	return(clf_probs)
