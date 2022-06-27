import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('C:/Users/CC-082/PycharmProjects/pythonProject/venv/DSINTERNSHIP/IRIS.csv')
logr=LogisticRegression(random_state=0)
gbm=GradientBoostingClassifier(n_estimators=10)
rfc=RandomForestClassifier(random_state=1)
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nn=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
mnb=MultinomialNB()

x=df.drop('species',axis=1)
y=df['species']
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

logr.fit(X_train,y_train)
logr_pred=logr.predict(X_test)

gbm.fit(X_train,y_train)
gbm_pred=gbm.predict(X_test)

rfc.fit(X_train,y_train)
rfc_pred=rfc.predict(X_test)

dt.fit(X_train,y_train)
dt_pred=dt.predict(X_test)

sv.fit(X_train,y_train)
sv_pred=sv.predict(X_test)

mnb.fit(X_train,y_train)
mnb_pred=mnb.predict(X_test)

nn.fit(X_train,y_train)
nn_pred=nn.predict(X_test)








print("Logistic Regression       :",accuracy_score(y_test,logr_pred))
print("GradientBoostingClassifier:",accuracy_score(y_test,gbm_pred))
print("RandomForestClassifier    :",accuracy_score(y_test,rfc_pred))
print("DecisionTreeClassifier    :",accuracy_score(y_test,dt_pred))
print("SVC                       :",accuracy_score(y_test,sv_pred))
print("MultinomialNB             :",accuracy_score(y_test,mnb_pred))
print("MLPClassifier             :",accuracy_score(y_test,nn_pred))



'''OUTPUT
Logistic Regression       : 0.9777777777777777
GradientBoostingClassifier: 0.9777777777777777
RandomForestClassifier    : 0.9777777777777777
DecisionTreeClassifier    : 0.9777777777777777
SVC                       : 0.9777777777777777
MultinomialNB             : 0.6
MLPClassifier             : 0.24444444444444444
'''