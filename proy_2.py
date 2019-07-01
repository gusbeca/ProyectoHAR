# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:46:36 2019

@author: Edgar Alberto Cañón
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


all_data = pd.DataFrame()

#Read data
path = 'C:/Users/Edgar Alberto Cañón/Documents/recognition/data.csv'

header = 'sensor_all_team'

sadl1 = pd.read_csv(path, sep=',', engine='python')
list(sadl1)
sadl1.dtypes

sadl1 = sadl1[['time', 'x', 'y', 'z', 'l']] 

data = sadl1.iloc[:, :243]
labels = sadl1.iloc[:,4]


columns =['x', 'y', 'z', 'l']

filtered_data=data

filtered_data['time_2']=pd.to_datetime(filtered_data['time'])
filtered_data.index=filtered_data.time_2

filtered_data = filtered_data.sort_index()
#calculate mean over a 1 secondwindow
means = filtered_data[columns].rolling('3s').mean()

keep = filtered_data.time_2.dt.microsecond/1000%500 
keep = keep -keep.shift() < 0

mode(['a','be','a','c'])

means = filtered_data[columns].rolling('3S').mean()[keep]
means.columns = [str(col) + '_mean' for col in means.columns]
variances = filtered_data[columns].rolling('3S').var()[keep]
variances.columns = [str(col) + '_var' for col in variances.columns]


labels.index = filtered_data.time_2
mode_labels = labels.rolling('3S').apply(lambda x: mode(x)[0])[keep]


all_features = pd.concat([means, variances], axis=1)
all_features['label'] = mode_labels
all_data = all_features

all_data = all_data.dropna()

list(all_data)


##### model

x = all_data[['x_mean',
 'y_mean',
 'z_mean',
 'x_var',
 'y_var',
 'z_var']]

y = all_data[['label']]

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)

clf.fit(X_train, y_train) 
y_predict=clf.predict(X_test)

#######confusion matrix

##classification report 
print(classification_report(y_test,y_predict))

#confusion matrix
print(confusion_matrix(y_test,y_predict))





