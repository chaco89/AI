dataset_filename = "training_data_new.csv"

# read file, put into dataset
import pandas as pd
dataset = pd.read_csv(dataset_filename)
x_train = dataset.iloc[:, 0].values;
y_train = dataset.iloc[:, 1].values;

# remove punctuation && double whitespace
from sklearn.feature_extraction.text import CountVectorizer
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(x_train)
#X_train_counts.shape


from sklearn.feature_extraction.text import TfidfTransformer
#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)
#X_train_tf.shape

#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#X_train_tfidf.shape

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=0,
                           max_iter=50, tol=None)),
])
text_clf.fit(x_train,y_train) 

# TESTING PREDICTIONNN
testing_dataset_filename = "testing_data_new.csv"
dataset_test = pd.read_csv(testing_dataset_filename)
x_test = dataset_test.iloc[:, 0].values;
y_test = dataset_test.iloc[:, 1].values;

predicted = text_clf.predict(x_test)
import numpy as np
np.mean(predicted == y_test)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(predicted, y_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print(f1_score(y_test, predicted, average="macro"))
print(precision_score(y_test, predicted, average="macro"))
print(recall_score(y_test, predicted, average="macro"))    

