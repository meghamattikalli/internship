# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:02:59 2018
accuracy is 0.902
roc-auc is 0.902
@author: HP
"""
import pandas as pd


# Importing the dataset
female= pd.read_csv('Indian-Female-Names1.csv')
female= female.apply(lambda x: x.astype(str).str.lower())
female=female.drop_duplicates(subset=['word'], keep='first')
female=female.iloc[:,0:1]
female["type"]=1

unwanted=['mr','.','moh.','@','miss','&','mrs','/','ku.','km',',','km0','-','(','smt','smt.','`','[']
for i in unwanted:
    female['word'] = female.word.str.replace(i, '')
numbers=['1','2','3','4','5','6','7','8','9','0']
for i in numbers:
    female['word'] = female.word.str.replace(i, '')
female['word'] = female.word.str.strip()
female['word'] = female.word.str.partition(" ")
female=female.drop_duplicates(subset=['word'], keep='first')
#preprocessing of names
#female['word'] = female.word.str.replace('miss', '')
#female['word'] = female.word.str.replace('mrs', '')
#female['word'] = female.word.str.replace('.', '')
#female['word'] = female.word.str.replace('smt', '')
#female['word'] = female.word.str.replace('@', '')
#female['word'] = female.word.str.strip()
#female['word'] = female.word.str.partition(" ")
#female=female.drop_duplicates(subset=['word'], keep='first')

female=female.head(2500)

male= pd.read_csv('Indian-Male-Names1.csv')
male= male.apply(lambda x: x.astype(str).str.lower())
male= male.drop_duplicates(subset=['word'], keep='first')
male=male.iloc[:,0:1]
male["type"]=1

for i in unwanted:
    male['word'] = male.word.str.replace(i, '')
for i in numbers:
    male['word'] = male.word.str.replace(i, '')
    
male['word'] = male.word.str.strip()
male['word'] = male.word.str.partition(" ")
male=male.drop_duplicates(subset=['word'], keep='first')

#male['word'] = male.word.str.replace('mr', '')
#male['word'] = male.word.str.replace('.', '')
#male['name'] = male.name.str.replace('smt', '')
#male['word'] = male.word.str.replace('@', '')
#male['word'] = male.word.str.strip()
#male['word'] = male.word.str.partition(" ")

male=male.head(2500)


#english words
eng_word = pd.read_csv('five_thousand.csv')
eng_word = eng_word.apply(lambda x: x.astype(str).str.lower())
eng_word['word'] = eng_word.word.str.replace('-', '')
eng_word['word'] = eng_word.word.str.replace("'", '')
eng_word['word'] = eng_word.word.str.replace("/", ' ')
eng_word['word'] = eng_word.word.str.partition(" ")


#eng_pre = pd.read_csv('prepositions.csv')
eng_word['type'] = 0
#eng_pre['type'] = 0


#concatenate names and english words
names=pd.concat([female,male,eng_word], axis=0)
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

for i in range(9,0,-1):
    zzz=len(df1[0])-i
    onehotencoder = OneHotEncoder(categorical_features = [zzz])
    df1 = onehotencoder.fit_transform(df1).toarray()
    df1 = df1[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1, y, test_size = 0.2, random_state = 0)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = len(df1[0])))

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

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
# Print model performance and plot the roc curve
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred)))
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred)))