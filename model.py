import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
data= pd.read_csv('C:\\Users\\jubyj\\Desktop\\main project\\heartdisease.csv')

data.drop(['education'],axis=1,inplace=True)
data['cigsPerDay']=data['cigsPerDay'].fillna(data['cigsPerDay'].median())
data.loc[(data['BPMeds'].isna()) & (data['sysBP'] <= 132), 'BPMeds'] = 0
data.loc[(data['BPMeds'].isna()) & (data['sysBP'] > 132), 'BPMeds'] = 1
data['totChol']=data['totChol'].fillna(data['totChol'].mean())
data.loc[(data['BMI'].isna()) & (data['diabetes'] == 0), 'BMI'] = 25.35
data.loc[(data['BMI'].isna()) & (data['diabetes'] == 1), 'BMI'] = 27.78
data.drop(689,inplace=True)
data.loc[(data['glucose'].isna()) & (data['diabetes'] == 0), 'glucose'] = 78
data.loc[(data['glucose'].isna()) & (data['diabetes'] == 1), 'glucose'] = 145

data['pulseBP']=data['sysBP']-data['diaBP']

y=data['TenYearCHD']
x=data.drop(['TenYearCHD','currentSmoker','sysBP','glucose'], axis =1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()
x_train=Scaler.fit_transform(x_train)
x_test=Scaler.fit_transform(x_test)
x_train=pd.DataFrame(x_train)

dtmodel=DecisionTreeClassifier()
dtmodel.fit(x_train,y_train)
y_pred=dtmodel.predict(x_test)

rf=RandomForestClassifier(max_depth=150, min_samples_leaf= 5, min_samples_split= 3 , n_estimators=300, random_state = 42)
rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
pickle.dump(dtmodel,open('model.pkl','wb'))
