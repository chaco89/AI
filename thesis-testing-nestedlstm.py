
saved_model = "final_nested_lstm_model.hd5"
training_dataset_filename = "training_data_new.csv"
testing_dataset_filename = "testing_data_new.csv"
dictionary_filename = "dictionary-nestedlstm-training.csv"

# load model
from nested_lstm import NestedLSTM
from keras.models import load_model
model = load_model(saved_model, custom_objects={'NestedLSTM': NestedLSTM})

from keras.utils import plot_model
plot_model(model, to_file='nested_lstm_layer.png', show_shapes=True, show_layer_names=True)

import json
with open('jsondata.txt') as json_file:  
    data = json.load(json_file)
    for p in data['lstm']:
        vocabSize = p['vocab_size']
        maxNumOfWord = p['max_num_of_word']

print(vocabSize)
print(maxNumOfWord)

# LOAD ONLY THE LABEL CLASSIFICATION FROM THE TRAINING, SO WE CAN KNOW THE CLASSIFICATION MAPPING
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset_train = pd.read_csv(training_dataset_filename)
y_train = dataset_train.iloc[:, 1].values;
label_encoder_training = LabelEncoder()
integer_encoded = label_encoder_training.fit_transform(y_train)

print(label_encoder_training.classes_)  
print(list(label_encoder_training.inverse_transform([0,1,2,3,4,5,6])))


# LOAD THE TESTING DATA
dataset_test = pd.read_csv(testing_dataset_filename)
x_test = dataset_test.iloc[:, 0].values;
y_test = dataset_test.iloc[:, 1].values;

import string
import re

#preprocess
df_testing = pd.DataFrame(data=x_test,columns=['sentence']);

for i in df_testing.index:
        translator = str.maketrans('', '', string.punctuation)
        df_testing.loc[i, "sentence"]= str.lower(re.sub(' +',' ', df_testing.loc[i,'sentence'].translate(translator))).split(); 


import csv 
model_dict = {}

with open(dictionary_filename, mode='r') as infile:
    reader = csv.reader(infile)
    model_dict = {rows[0]:rows[1] for rows in reader}
      
      
encoded_sentence = []
for i in df_testing.index:
    sentence = df_testing.loc[i, "sentence"]
    idx = 0
    for word in sentence:
        try:
            sentence[idx] = model_dict[word]
        except KeyError: 
            sentence[idx] = 0
        idx = idx + 1
    encoded_sentence.append(sentence)

from keras.preprocessing.sequence import pad_sequences
padded_sentences_test = pad_sequences(encoded_sentence, maxlen=maxNumOfWord, padding='pre')

# ENCODE THE LABEL CLASSES
from sklearn.preprocessing import LabelEncoder
label_encoder_testing = LabelEncoder()
integer_encoded_testing = label_encoder_testing.fit_transform(y_test)

# PREDICT THE SENTENCES WHICH BELONG INTO CLASSIFICATION
y_predicted = model.predict_classes(padded_sentences_test)

### PRINT INTO CSV ###
df_result = pd.DataFrame(columns=['sentence', 'actual', 'predicted'])

dataset_test = pd.read_csv(testing_dataset_filename)
dummy_x_test = dataset_test.iloc[:, 0].values;
dummy_y_test = dataset_test.iloc[:, 1].values;

#print(dummy_x_test[0])
#print(dummy_y_test[0])
#print(y_predicted[0])
#print(list(label_encoder_testing.inverse_transform([y_predicted[0]])))

for i in range(len(dummy_x_test)):
    sentence = dummy_x_test[i]
    actual_emotion = dummy_y_test[i]
    inverse_transform_emotion = list(label_encoder_testing.inverse_transform([y_predicted[i]]))
    predicted_emotion = ''.join(inverse_transform_emotion)
    df_result.loc[i] = [dummy_x_test[i], dummy_y_test[i], predicted_emotion]
    
df_result.to_excel("result_details_excel.xlsx")
### END OF PRINT INTO CSV ###


# GENERATE THE CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
matrix =confusion_matrix(integer_encoded_testing, y_predicted)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print(f1_score(integer_encoded_testing, y_predicted, average="macro"))
print(precision_score(integer_encoded_testing, y_predicted, average="macro"))
print(recall_score(integer_encoded_testing, y_predicted, average="macro"))    


with open("matrix.csv", "w", newline='') as resultFile:
    writer = csv.writer(resultFile, dialect='excel')
    writer.writerows(matrix)



print(label_encoder_testing.classes_)  
print(list(label_encoder_testing.inverse_transform([0,1,2,3,4,5,6])))

