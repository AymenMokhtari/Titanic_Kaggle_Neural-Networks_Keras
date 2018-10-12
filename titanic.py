# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:59:29 2018
@author: Aymen
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Imputer




dataset = pd.read_csv('train.csv')

#(thresh=1) itearates through all the rows and keeps each row that has at least 30% non-na value
X = dataset.dropna(thresh=0.3*len(dataset), axis=1)


X =  X.drop('Survived',axis=1)
X =  X.drop('Ticket',axis=1)
y = dataset['Survived']
y = y.values

#number of missing values of each column
print(X.isnull().sum())


#percentage of missing values of each column

print(X.isnull().sum()* 100 / len(X))
#
X.isnull().mean()


#remove cabin colum because it has more than 60% missing values
#X =X[X.columns[X.isnull().mean() < 0.6]]

#replace missing value of column age with mean
print(X["Age"].mean())
X['Age'] = X["Age"].replace(np.NaN , X["Age"].mean())

#check missing values 
print(X.isnull().sum()* 100 / len(X))

#replace  0 with NaN it is not the case here
#X = X.replace(0, np.NaN)


#predict missing value 

#X =  X.drop('Name',axis=1)
#X =  X.drop('Ticket',axis=1)
#X =  X.drop('SibSp',axis=1)
#X =  X.drop('Embarked',axis=1)


#dealing with missing values


X["Embarked"] = X["Embarked"].fillna("U")

#check missing values  again
print(X.isnull().sum()* 100 / len(X))



# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def create_dummies(df,column_name):
   dummies = pd.get_dummies(df[column_name],prefix=column_name)
   df = pd.concat([df,dummies],axis=1)
   return df

titles = {
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"
}


extracted_titles = X["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
X["Title"] = extracted_titles.map(titles)
le = LabelEncoder()


for column in ["Sex","Embarked","Title"]: #Columns who diveded
    X = create_dummies(X,column)
    
    
    
X = X.drop(['Sex','Embarked','Title','Name'], axis=1) 

'''
#or do it separately
for column in ["Sex"]:
    X = create_dummies(X,column)
'''






'''
le.fit(X['Embarked'].astype(str))
X['Embarked'] = le.transform(X['Embarked'].astype(str))
'''



# feature selection 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd




    
#plot feature importance chart

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(X, y)
features = pd.DataFrame()
features['feature'] = X.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(25, 5))



#remove not important features "embarked U"  "title_royality" 

X = X.drop(['Embarked_U','Title_Royalty'], axis=1) 


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization


classifier = Sequential()




classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
classifier.add(Dropout(p = 0.5))
classifier.add(BatchNormalization())




classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(BatchNormalization())

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(BatchNormalization())








classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.callbacks import TensorBoard

from time import time

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))





history =classifier.fit(X_train, y_train, nb_epoch=300, batch_size=16, validation_data=(X_test, y_test), callbacks=[tensorboard])






plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



scoretrain = classifier.evaluate(X_train ,y_train , batch_size=10)
scoretest = classifier.evaluate(X_test ,y_test , batch_size=10)

print (scoretrain)
print (scoretest)

#end


test_df = pd.read_csv('test.csv')
PassengerId = test_df['PassengerId']
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def create_dummies(df,column_name):
   dummies = pd.get_dummies(df[column_name],prefix=column_name)
   df = pd.concat([df,dummies],axis=1)
   return df

titles = {
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"
}


extracted_titles = test_df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
test_df["Title"] = extracted_titles.map(titles)
le = LabelEncoder()


for column in ["Sex","Embarked","Title"]: #Columns who diveded
    test_df = create_dummies(test_df,column)
    
    
    
test_df = test_df.drop(['Sex','Embarked','Title','Name','Cabin','PassengerId','Ticket'], axis=1) 
test_predict = classifier.predict(test_df)
test_predict = np.where(test_predict>0.5,1,0)
NNSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': test_predict.ravel() })
NNSubmission.to_csv("NNSubmission.csv", index=False)




'''


end here





'''

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
'''
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
'''




callbacks = [EarlyStopping(monitor='val_loss', patience=100),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

history =classifier.fit(X_train, y_train, nb_epoch=50, batch_size=10, validation_data=(X_test, y_test) , callbacks= callbacks)


score = classifier.evaluate(X_test, y_test, verbose=True) 
scoret = classifier.evaluate(X_train, y_train, verbose=False) 


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 1000)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()




from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [ 50,22, 32 ,10],
              'epochs': [100, 200, 300 ],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


