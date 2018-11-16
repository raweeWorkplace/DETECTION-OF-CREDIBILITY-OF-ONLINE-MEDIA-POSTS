from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import load_data as ld
import text_tokenizing as tt
import embedding_layer as emb_layer
import test_data as td

#Some constant values
maxlen = 100  # We will cut reviews after 50 words
max_words = 50000  # We will only consider the top 50,000 words in the dataset


#labeling the data set
texts,labels = ld.data_label()

#sampling the text data
x_train,y_train,x_val,y_val,word_index = tt.tokenize(texts,labels,maxlen,max_words)


clf = RandomForestClassifier(n_estimators=25)
clf.fit(x_train, y_train)
x_test,y_test = td.load_test_data()
clf_probs = clf.predict(x_test)
print(accuracy_score(clf_probs,y_test))