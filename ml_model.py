# importing all the libraries 
import numpy as np 
import pandas as pd
from sklearn import ensemble 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib 

print("All required modules are imported succesfully ")

# this will read the databse from the csv format 
df = pd.read_csv('kidney_disease.csv') 

df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
df.rename(columns={'classification':'class'},inplace=True)
df['pe'] = df['pe'].replace(to_replace='good',value=0)
df['appet'] = df['appet'].replace(to_replace='no',value=0)
df['cad'] = df['cad'].replace(to_replace='\tno',value=0)
df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

df.drop('id',axis=1,inplace=True)
df = df.dropna(axis=0)

cols = ['bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc']
X = df[cols]
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=44, stratify= y)
print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

# Creating the machine learning model in python 

# creaating an Random Forest Classifier for our machine learning model
model = ensemble.RandomForestClassifier()

# fitting the data to the machine learing model for training ( this is the training process of the ml model )
model.fit(X_train , y_train)

# getting the predictions from the machine learning model 
y_pred = model.predict(X_test) 

# this will print the accuracy score of the machine learning 
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

# this is the classification report of the machine learning model 
clf_report = classification_report(y_test, y_pred) 

print("Classification Report of the ML Model ")
print("----------------------------------------")
print(clf_report)
print("----------------------------------------")

# this is the codeto dumo and create an object of the machine learning model to be used later to make the API 
joblib.dump(model, "kidney_model.pkl")
print("Object File is created in the project working directory")

