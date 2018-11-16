from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import tensorflow as tf
import test_data as td
# load json and create model
json_file = open('model21.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("pre_trained_glove_model.h5")
print("Loaded model from disk")

# text = ["Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in."]
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(text)
# sequences = tokenizer.texts_to_sequences(text)
# session = tf.Session()

# data = pad_sequences(sequences,maxlen=100)
# print(data)

# score = model.predict(data)
# print(score)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
x_test,y_test = td.load_test_data()
clf_probs = model.evaluate(x_test,y_test)
print(clf_probs)