import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df=pd.read_excel('StudentInfo1.xlsx')
df=df[['Start Date','Gender','Registration date','Active','Starting Age','Starting Grade','BlackBelt']]
df=df.dropna()
df['Gender'] = df['Gender'].map( {'F': 0, 'M': 1} )

#####################################################################
#Random Forest
X=df[['Gender','Starting Age','Starting Grade']]
y=df['BlackBelt']
X.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test

RF=RandomForestClassifier()
RF=RF.fit(X_train,y_train)
y_preRF =RF.predict(X_test)

############################################################################################
Gender='M'
Age=8
Grade=10

if Gender=='M':
    Gender=1
else:
    Gender=0

Xnew={'Gender': [Gender], 'Starting Age':[Age],'Starting Grade':[Grade]}
Xnew=pd.DataFrame(Xnew)
yRF=RF.predict_proba(Xnew)[:,1]
#print('The probability of achieving a black belt: %.2f'% ((yRF)*100),'%')
print('The probability of achieving a black belt: ')
print('Random Forest : %.2f'% ((yRF)*100),'%')
############################################################################################


RF_PKL = open('RF.pickle','wb')
pickle.dump(RF,RF_PKL)