# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing libraries
import pandas as pd
import numpy as np

# import dataset
df = pd.read_csv('train.csv')
df.info()

df.columns
train_set = df.drop(['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature','GarageYrBlt','GarageCars','OverallQual','SalePrice'], axis = 1)

for i in train_set.iloc[:,:].columns:
   print( train_set[i].unique())

# correlation and heatmap
corr = train_set.corr()

import seaborn as sns

# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)


X = train_set.iloc[:,:].values
y = df.iloc[:,-1].values

X[pd.isnull(X)] = 'NaN'
X = pd.DataFrame(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X.iloc[:,1] = le.fit_transform(X.iloc[:,1])
X.iloc[:,4] = le.fit_transform(X.iloc[:,4])
X.iloc[:,5] = le.fit_transform(X.iloc[:,5])
X.iloc[:,6] = le.fit_transform(X.iloc[:,6])
X.iloc[:,7] = le.fit_transform(X.iloc[:,7])
X.iloc[:,8] = le.fit_transform(X.iloc[:,8])
X.iloc[:,9] = le.fit_transform(X.iloc[:,9])
X.iloc[:,10] = le.fit_transform(X.iloc[:,10])
X.iloc[:,11] = le.fit_transform(X.iloc[:,11])
X.iloc[:,12] = le.fit_transform(X.iloc[:,12])
X.iloc[:,13] = le.fit_transform(X.iloc[:,13])
X.iloc[:,14] = le.fit_transform(X.iloc[:,14])
X.iloc[:,18] = le.fit_transform(X.iloc[:,18])
X.iloc[:,19] = le.fit_transform(X.iloc[:,19])
X.iloc[:,20] = le.fit_transform(X.iloc[:,20])
X.iloc[:,21] = le.fit_transform(X.iloc[:,21])
X.iloc[:,22] = le.fit_transform(X.iloc[:,22])
X.iloc[:,24] = le.fit_transform(X.iloc[:,24])
X.iloc[:,25] = le.fit_transform(X.iloc[:,25])
X.iloc[:,26] = le.fit_transform(X.iloc[:,26])
X.iloc[:,27] = le.fit_transform(X.iloc[:,27])
X.iloc[:,28] = le.fit_transform(X.iloc[:,28])
X.iloc[:,29] = le.fit_transform(X.iloc[:,29])
X.iloc[:,30] = le.fit_transform(X.iloc[:,30])
X.iloc[:,32] = le.fit_transform(X.iloc[:,32])
X.iloc[:,36] = le.fit_transform(X.iloc[:,36])
X.iloc[:,37] = le.fit_transform(X.iloc[:,37])
X.iloc[:,38] = le.fit_transform(X.iloc[:,38])
X.iloc[:,39] = le.fit_transform(X.iloc[:,39])
X.iloc[:,50] = le.fit_transform(X.iloc[:,50])
X.iloc[:,52] = le.fit_transform(X.iloc[:,52])
X.iloc[:,54] = le.fit_transform(X.iloc[:,54])
X.iloc[:,55] = le.fit_transform(X.iloc[:,55])
X.iloc[:,57] = le.fit_transform(X.iloc[:,57])
X.iloc[:,58] = le.fit_transform(X.iloc[:,58])
X.iloc[:,59] = le.fit_transform(X.iloc[:,59])
X.iloc[:,69] = le.fit_transform(X.iloc[:,69])
X.iloc[:,70] = le.fit_transform(X.iloc[:,70])


















# =============================================================================
  col = []
  for i in range(1,len(test_set)+1):
      for k in range(72):
          if test_set.iloc[i:i+1,k:k+1].values == 'NaN':
              col.append(k)
              
  
  col = pd.DataFrame(col)
# =============================================================================
col.iloc[:,0].unique()



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
X.iloc[:,2:3] = imputer.fit_transform(X.iloc[:,2:3])
X.iloc[:,22:23] = imputer.fit_transform(X.iloc[:,22:23])
X.iloc[:,23:24] = imputer.fit_transform(X.iloc[:,23:24])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import SGD

classifier = Sequential()

classifier.add(Dense(output_dim = 512, activation = 'relu', input_dim = 71))
classifier.add(Dropout(0.25))
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 1, activation = 'relu'))

classifier.compile(optimizer = 'adam',loss = 'mean_absolute_error', metrics=['mae']) 

history = classifier.fit(X_train, y_train, batch_size = 32, epochs = 150, validation_data = (X_test, y_test), verbose = 1)



score = classifier.evaluate(X_test, y_test, verbose = 0)
print('Test loss: ',score[0])
print('Test accuracy',score[1])

prediction = classifier.predict(X_test)

import matplotlib.pyplot as plt
plt.plot(y_test,color = 'red')
plt.plot(prediction, color = 'blue')
plt.show()


# ANN Tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 512, activation = 'relu', input_dim = 71))
    classifier.add(Dropout(0.25))
    classifier.add(BatchNormalization())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(output_dim = 1, activation = 'relu'))
    classifier.compile(optimizer = optimizer,loss = 'mean_absolute_error', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build)
parameters = {'batch_size' : [32, 128],
              'epochs' : [150, 500],
              'optimizer' : ['adam', SGD(0.01)]
              }
 
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_

# importing test_set and arrange
test_set = pd.read_csv('test.csv')
test_set = test_set.drop(['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature','GarageYrBlt','GarageCars','OverallQual'], axis = 1)
test_set = test_set.iloc[:,:].values
test_set[pd.isnull(test_set)] = 'NaN'

test_set = pd.DataFrame(test_set)

test_set.iloc[:,1] = le.fit_transform(test_set.iloc[:,1])
test_set.iloc[:,4] = le.fit_transform(test_set.iloc[:,4])
test_set.iloc[:,5] = le.fit_transform(test_set.iloc[:,5])
test_set.iloc[:,6] = le.fit_transform(test_set.iloc[:,6])
test_set.iloc[:,7] = le.fit_transform(test_set.iloc[:,7])
test_set.iloc[:,8] = le.fit_transform(test_set.iloc[:,8])
test_set.iloc[:,9] = le.fit_transform(test_set.iloc[:,9])
test_set.iloc[:,10] = le.fit_transform(test_set.iloc[:,10])
test_set.iloc[:,11] = le.fit_transform(test_set.iloc[:,11])
test_set.iloc[:,12] = le.fit_transform(test_set.iloc[:,12])
test_set.iloc[:,13] = le.fit_transform(test_set.iloc[:,13])
test_set.iloc[:,14] = le.fit_transform(test_set.iloc[:,14])
test_set.iloc[:,18] = le.fit_transform(test_set.iloc[:,18])
test_set.iloc[:,19] = le.fit_transform(test_set.iloc[:,19])
test_set.iloc[:,20] = le.fit_transform(test_set.iloc[:,20])
test_set.iloc[:,21] = le.fit_transform(test_set.iloc[:,21])
test_set.iloc[:,22] = le.fit_transform(test_set.iloc[:,22])
test_set.iloc[:,24] = le.fit_transform(test_set.iloc[:,24])
test_set.iloc[:,25] = le.fit_transform(test_set.iloc[:,25])
test_set.iloc[:,26] = le.fit_transform(test_set.iloc[:,26])
test_set.iloc[:,27] = le.fit_transform(test_set.iloc[:,27])
test_set.iloc[:,28] = le.fit_transform(test_set.iloc[:,28])
test_set.iloc[:,29] = le.fit_transform(test_set.iloc[:,29])
test_set.iloc[:,30] = le.fit_transform(test_set.iloc[:,30])
test_set.iloc[:,32] = le.fit_transform(test_set.iloc[:,32])
test_set.iloc[:,36] = le.fit_transform(test_set.iloc[:,36])
test_set.iloc[:,37] = le.fit_transform(test_set.iloc[:,37])
test_set.iloc[:,38] = le.fit_transform(test_set.iloc[:,38])
test_set.iloc[:,39] = le.fit_transform(test_set.iloc[:,39])
test_set.iloc[:,50] = le.fit_transform(test_set.iloc[:,50])
test_set.iloc[:,52] = le.fit_transform(test_set.iloc[:,52])
test_set.iloc[:,54] = le.fit_transform(test_set.iloc[:,54])
test_set.iloc[:,55] = le.fit_transform(test_set.iloc[:,55])
test_set.iloc[:,57] = le.fit_transform(test_set.iloc[:,57])
test_set.iloc[:,58] = le.fit_transform(test_set.iloc[:,58])
test_set.iloc[:,59] = le.fit_transform(test_set.iloc[:,59])
test_set.iloc[:,69] = le.fit_transform(test_set.iloc[:,69])
test_set.iloc[:,70] = le.fit_transform(test_set.iloc[:,70])

test_set.iloc[:,2:3] = imputer.fit_transform(test_set.iloc[:,2:3])
test_set.iloc[:,22:23] = imputer.fit_transform(test_set.iloc[:,22:23])
test_set.iloc[:,23:24] = imputer.fit_transform(test_set.iloc[:,23:24])
test_set.iloc[:,31:32] = imputer.fit_transform(test_set.iloc[:,31:32])
test_set.iloc[:,33:34] = imputer.fit_transform(test_set.iloc[:,33:34])
test_set.iloc[:,34:35] = imputer.fit_transform(test_set.iloc[:,34:35])
test_set.iloc[:,35:36] = imputer.fit_transform(test_set.iloc[:,35:36])
test_set.iloc[:,44:45] = imputer.fit_transform(test_set.iloc[:,44:45])
test_set.iloc[:,45:46] = imputer.fit_transform(test_set.iloc[:,45:46])
test_set.iloc[:,56:57] = imputer.fit_transform(test_set.iloc[:,56:57])




test_set = test_set.astype('float32')
test_set = sc.fit_transform(test_set)


submission = classifier.predict(test_set)

plt.plot(submission)
plt.show()

sample = pd.read_csv('sample_submission.csv')

Id = []
for i in range(1461,2920):
    Id.append(i)
Id = pd.DataFrame(Id)
submission = pd.DataFrame(submission)

submission = pd.concat([Id,submission],axis = 1)
submission = pd.DataFrame(submission.iloc[:,:].values, columns = ['Id','SalePrice'])
submission = submission.astype('int32')

submission.to_csv(r'submission1.csv',index = False)

