


import numpy as np
import pandas as pd 
from matplotlib import pyplot as pt
data=pd.read_csv('data_final.csv')
data.head(1)

# dividing data
x=data.iloc[: , 2:].values
y=data.iloc[: , 1].values

#let do encoding for y data 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder=LabelEncoder()
y  =  encoder.fit_transform(y])
#one=OneHotEncoder(categorical_features=[0])
#y=one.fit_transform(y).toarray()


#spliting data into traning or other things
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=0)



#import clasifier 

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)

classifier.fit(x_train,y_train)

predict=classifier.predict(x_test)
#predicting result
 print(predict)

