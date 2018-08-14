# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:58:07 2018

@author: HP
"""

qstr="बhe"
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 18:18:15 2018

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 22:33:07 2018

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

# Importing the dataset
female= pd.read_csv('Indian-Female-Names.csv')
female=female.iloc[:,0:1]
female["type"]=1

male= pd.read_csv('Indian-Male-Names.csv')
male=male.iloc[:,0:1]
male["type"]=1

human_names=pd.concat([female,male], axis=0)
human_names= human_names.apply(lambda x: x.astype(str).str.lower())
human_names=human_names.drop_duplicates(subset=['word'], keep='first')

unwanted=['mr','.','moh.','@','miss','&','mrs','/','ku.','km',',','km0','-','(','smt','smt.','`','[']
for i in unwanted:
    human_names['word'] = human_names.word.str.replace(i, '')
numbers=['1','2','3','4','5','6','7','8','9','0']
for i in numbers:
    human_names['word'] = human_names.word.str.replace(i, '')
human_names['word'] = human_names.word.str.strip()
human_names['word'] = human_names.word.str.partition(" ")
human_names=human_names.drop_duplicates(subset=['word'], keep='first')

human_names=human_names.sort_values(by=['word'])
human_names.index=[i for i in range(0,len(human_names.index))] #renaming the indices
human_names=human_names.iloc[0:6598,:]
#male=male.head(100)


#english words
eng_word= pd.read_csv('common_thousand.csv')
eng_pre=pd.read_csv('prepositions.csv')
eng_word['type']=0
eng_pre['type']=0


#concatenate names and english words
names=pd.concat([human_names,eng_word,eng_pre], axis=0)
names=names.dropna()


names.index=[i for i in range(0,len(names.index))] #renaming the indices

X=names['word'] #features
y=names['type'] #labels
        
#padding and truncating X
for i in range(0,len(X.index)):
    if(len(X.loc[i])>10):
        X.loc[i]=X.loc[i][0:10]
    else:    
        X.loc[i]=X.loc[i].ljust(10) 

def letter_to_int(letter):
    alphabet = list('abcdefghijklmnopqrstuvwxyz ')
    return alphabet.index(letter)


list1=[]
for j in range(0,len(X.index)):
    z=[]
    for i in X.loc[j]:
        z.append(letter_to_int(i))
    list1.append(tuple(z))
     
    
df=pd.DataFrame(list1)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
df1 = onehotencoder.fit_transform(df).toarray()
df1 = df1[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [25])
df1 = onehotencoder.fit_transform(df1).toarray()
df1 = df1[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [51])
df1 = onehotencoder.fit_transform(df1).toarray()
df1 = df1[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [77])
df1 = onehotencoder.fit_transform(df1).toarray()
df1 = df1[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [102])
df1 = onehotencoder.fit_transform(df1).toarray()
df1 = df1[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [128])
df1 = onehotencoder.fit_transform(df1).toarray()
df1 = df1[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [154])
df1 = onehotencoder.fit_transform(df1).toarray()
df1 = df1[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [178])
df1 = onehotencoder.fit_transform(df1).toarray()
df1 = df1[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [203])
df1 = onehotencoder.fit_transform(df1).toarray()
df1 = df1[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [226])
df1 = onehotencoder.fit_transform(df1).toarray()
df1 = df1[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1, y, test_size = 0.2, random_state = 0)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 249))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred=pd.DataFrame(y_pred)
y_pred=y_pred.iloc[:,0]
for i in range(0,len(y_pred)):
    if y_pred.loc[i]==True:
        y_pred.loc[i]=1
    else:
        y_pred.loc[i]=0

for i in range(0,len(y_pred)):
    y_pred.loc[i]=y_pred.loc[i].item()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
# Print model performance and plot the roc curve
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred)))