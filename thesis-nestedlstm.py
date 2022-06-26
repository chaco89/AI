total_epoch = 50
dataset_filename = "training_data_new.csv"
dictionary_filename = "dictionary-nestedlstm-training.csv"
checkpoint_filepath="nestedlstm-weights.{epoch:02d}-{acc:.3f}-{loss:.3f}.hdf5"
saved_model_name = "final_nested_lstm_model.hd5"

# read file, put into dataset
import pandas as pd
dataset = pd.read_csv(dataset_filename)
x_train = dataset.iloc[:, 0].values;
y_train = dataset.iloc[:, 1].values;

df = pd.DataFrame(data=x_train,columns=['sentence']);

# remove punctuation && double whitespace
import numpy as np
df_split = np.array_split(df, 1000)

import string
import re

vocabSize = 0
dictionary = set()
maxNumOfWord = 0

for dataframe in df_split:
    for i in dataframe.index:
        translator = str.maketrans('', '', string.punctuation)
        dataframe.loc[i, "sentence"]= str.lower(re.sub(' +',' ', dataframe.loc[i,'sentence'].translate(translator))).split(); 
        sentence = dataframe.loc[i,'sentence']        
        numOfWord = len(sentence)
        
        for word in sentence:
           if word not in dictionary : 
                dictionary.add(word)
                vocabSize = vocabSize + 1
            
        if maxNumOfWord < numOfWord :
            maxNumOfWord = numOfWord

print(maxNumOfWord)
print(vocabSize)       

import json
jsonData = {'lstm':[{'filename': dataset_filename, 'max_num_of_word': maxNumOfWord, 'vocab_size': vocabSize }]}

with open('jsondata.txt', 'w') as outfile:  
    json.dump(jsonData, outfile)

import csv

dictionary = list(dictionary)
model_dict = {}

i = 1
for word in dictionary:
        model_dict[word] = i
        i = i + 1           

with open(dictionary_filename,'w', newline='') as resultFile:
    w = csv.writer(resultFile)
    w.writerows(model_dict.items())

dfList = []
for dataframe in df_split:
    dfList.append(dataframe)
df = pd.concat(dfList, ignore_index=True)        
    
df.to_csv("sentence.csv", sep=',', encoding='utf-8')

encoded_sentence = []
for dataframe in df_split:
    for i in dataframe.index:
        sentence = dataframe.loc[i, "sentence"]
        idx = 0
        for word in sentence:
            sentence[idx] = model_dict[word]
            idx = idx + 1
        encoded_sentence.append(sentence)
        
# to make sure mapping on training and testing in dataset is same using hashing_trick    
with open("encoded_sentence.csv", "w", newline='') as resultFile:
    writer = csv.writer(resultFile, dialect='excel')
    writer.writerows(encoded_sentence)

from keras.preprocessing.sequence import pad_sequences
padded_sentences = pad_sequences(encoded_sentence, maxlen=maxNumOfWord, padding='pre')

# encode emotion to integer
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
print(list(label_encoder.inverse_transform([0,1,2,3,4,5,6])))
   
# create Nested LSTM model	
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from nested_lstm import NestedLSTM

model = Sequential();
model.add(Embedding(vocabSize, 50, input_length = maxNumOfWord))
model.add(NestedLSTM(50, depth=2, dropout=0.2, recurrent_dropout=0.0))
model.add(Dense(7, activation='softmax'))
optimizer = Adam(lr=0.00001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#print(model.summary())

# checkpoint
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# fit the model
model.fit(padded_sentences, integer_encoded, epochs=50, batch_size= 100, callbacks=callbacks_list)

# evaluate the model
loss, accuracy = model.evaluate(padded_sentences, integer_encoded)
print('Accuracy: %f' % (accuracy*100))

# save model
model.save(saved_model_name)