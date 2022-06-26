dataset_filename = "testing_data_new.csv"

# read file, put into dataset
import pandas as pd
dataset = pd.read_csv(dataset_filename)
x_train = dataset.iloc[:, 0].values;
y_train = dataset.iloc[:, 1].values;

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words = None)
X_train_counts = count_vect.fit_transform(x_train)

import csv
with open("X_train_counts.csv", "w", newline='') as resultFile:
    writer = csv.writer(resultFile, dialect='excel')
    writer.writerows(X_train_counts.toarray())

print(X_train_counts[0].toarray())
print(count_vect.get_feature_names());    
print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.toarray())
print(X_train_tf.shape)

from sklearn.linear_model import SGDClassifier
text_clf = SGDClassifier(random_state=0).fit(X_train_tf, y_train)

# TESTING
testing_dataset_filename = "testing_data_new.csv"
dataset_test = pd.read_csv(testing_dataset_filename)
x_test = dataset_test.iloc[:, 0].values;
y_test = dataset_test.iloc[:, 1].values;

X_test_counts = count_vect.transform(x_test)
print(X_test_counts.shape);
X_test_tf = tf_transformer.transform(X_test_counts)
predicted = text_clf.predict(X_test_tf)


import numpy as np
np.mean(predicted == y_test)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(predicted, y_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print(f1_score(y_test, predicted, average="macro"))
print(precision_score(y_test, predicted, average="macro"))
print(recall_score(y_test, predicted, average="macro"))    
