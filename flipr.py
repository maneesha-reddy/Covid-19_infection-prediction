import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

#dataset reading
import csv

data1=pd.read_csv("C:/Users/Dell/Desktop/flipr/Test_dataset.csv", encoding='latin-1')
obj_df=data1
#print(obj_df)
df=pd.DataFrame(obj_df)
#print(df.isnull())
obj_df = obj_df.fillna({"cardiological pressure": "Normal"})
obj_df=pd.get_dummies(obj_df, columns=["Gender", "Region", "Designation", "Married",
      "Occupation", "Mode_transport","comorbidity", "Pulmonary score",
       "cardiological pressure",], prefix=["Gender", "region","designation","married","occupation",
        "mode_transport","comorbidity","pulmonary score","cardiological pressure"])
standardScaler = StandardScaler()
columns_to_scale = ["Children","cases/1M", "Deaths/1M","Age", "Coma score", "Diuresis", "Platelets", "HBB","d-dimer",
       "Heart rate", "HDL cholesterol", "Charlson Index", "Blood Glucose",
       "Insurance", "salary", "FT/month"]
obj_df[columns_to_scale] = standardScaler.fit_transform(obj_df[columns_to_scale])
obj_df["Name"]=obj_df["Name"]
obj_df["people_ID"]=obj_df["people_ID"]
region_Bengaluru=[0]*obj_df.shape[0]
obj_df['region_Bengaluru'] = region_Bengaluru

region_Bhubaneshwar=[0]*obj_df.shape[0]
obj_df['region_Bhubaneshwar'] = region_Bhubaneshwar

region_Chandigarh=[0]*obj_df.shape[0]
obj_df['region_Chandigarh'] = region_Chandigarh

region_Chennai=[0]*obj_df.shape[0]
obj_df['region_Chennai'] = region_Chennai
peopleID=obj_df['people_ID']
#print("people")
#print(peopleID)
Xnew= obj_df.drop(['people_ID','Insurance','Name'], axis = 1)



#print(X[X.isnull().any(axis=1)])
Xnew=Xnew.fillna(method ='pad')
#print(Xnew)

def dataset(data):
    obj_df=data
    #print(obj_df)
    #print(data)
    #print(data.columns)
    df=pd.DataFrame(obj_df)
    #print(df.isnull())

    obj_df = obj_df.fillna({"cardiological pressure": "Normal"})
    obj_df=pd.get_dummies(obj_df, columns=["Gender", "Region", "Designation", "Married",
          "Occupation", "Mode_transport","comorbidity", "Pulmonary score",
           "cardiological pressure",], prefix=["Gender", "region","designation","married","occupation",
            "mode_transport","comorbidity","pulmonary score","cardiological pressure"])
    #print(obj_df)
    standardScaler = StandardScaler()
    columns_to_scale = ["Children","cases/1M", "Deaths/1M","Age", "Coma score", "Diuresis", "Platelets", "HBB","d-dimer",
           "Heart rate", "HDL cholesterol", "Charlson Index", "Blood Glucose",
           "Insurance", "salary", "FT/month"]
    obj_df[columns_to_scale] = standardScaler.fit_transform(obj_df[columns_to_scale])
    obj_df["Name"]=obj_df["Name"]
    obj_df["people_ID"]=obj_df["people_ID"]
    obj_df["Infect_Prob"]=obj_df["Infect_Prob"]
    region_hyd=[0]*obj_df.shape[0]
    obj_df['region_Hyderabad'] = region_hyd
    region_Kolkata=[0]*obj_df.shape[0]
    obj_df['region_Kolkata'] = region_Kolkata
    region_Mumbai=[0]*obj_df.shape[0]
    obj_df['region_Mumbai'] = region_Mumbai
    region_Pune=[0]*obj_df.shape[0]
    obj_df['region_Pune'] = region_Pune
    region_Thiruvananthapuram=[0]*obj_df.shape[0]
    obj_df['region_Thiruvananthapuram'] = region_Thiruvananthapuram
    y = obj_df['Infect_Prob']
    X = obj_df.drop(['Infect_Prob','people_ID','Insurance','Name'], axis = 1)
    X=X.fillna(method ='pad')
    return X,y


data=pd.read_csv("C:/Users/Dell/Desktop/flipr/Train_dataset.csv", encoding='latin-1')

X,y=dataset(data)
y = preprocessing.Binarizer(50).fit_transform(pd.DataFrame(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
#X = obj_df.drop(['Infect_Prob','people_ID','Insurance','Name'], axis = 1)
rf_classifier = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf_classifier.fit(X_train, y_train)
predictions1 = rf_classifier.predict(X_test)
#print(predictions1[:,None])
predictions=rf_classifier.predict_proba(X_test)
#print(predictions)
print("Accuracy: ",rf_classifier.score(X_test, y_test))
#print(y_test)
roc_value = roc_auc_score(y_test, predictions1[:,None])
print(roc_value)

ynew=rf_classifier.predict_proba(Xnew)
#print(ynew[:,1])
output = pd.DataFrame(peopleID)
output["Infect_Prob"] = ynew[:,1]*100
export_csv = output.to_csv (r"C:/Users/Dell/Desktop/output.csv", index = None, header=True)
