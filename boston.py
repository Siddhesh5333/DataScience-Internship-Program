import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df=pd.read_csv('C:/Users/CC-110/PycharmProjects/pythonProject/housing.csv',header=None,delimiter=r"\s+",names=column_names)


for col in column_names:
    if df[col].count()!=506:
        df[col].fillna(df[col].median(),inplace=True)

# print(mean_squared_error(y_test,y_pred))

#Dropping CHAS because it has discrete values
df=df.drop('CHAS',axis=1)

# Box plot diagram
'''
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()
'''

#Outliers percentage in every column
'''
for k, v in df.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))
'''
#Outliers Results
'''
Column CRIM outliers = 13.04%
Column ZN outliers = 13.44%
Column INDUS outliers = 0.00%
Column NOX outliers = 0.00%
Column RM outliers = 5.93%
Column AGE outliers = 0.00%
Column DIS outliers = 0.99%
Column RAD outliers = 0.00%
Column TAX outliers = 0.00%
Column PTRATIO outliers = 2.96%
Column B outliers = 15.22%
Column LSTAT outliers = 1.38%
Column MEDV outliers = 7.91%
'''


#Dropping Outliers greater than 10%
df=df.drop('CRIM',axis=1)
df=df.drop('ZN',axis=1)
df=df.drop('B',axis=1)

# MEDV max value is greater than 50.Based on that, values above 50.00 may not help to predict MEDV
# lets remove MEDV value above 50
df= df[~(df['MEDV'] >= 50.0)]

#Instances
lreg=LinearRegression()
rf=RandomForestRegressor(random_state=0)
gb=GradientBoostingRegressor(n_estimators=10)
dc=DecisionTreeRegressor(random_state=0)
sv=SVR()
mlp=MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)
gnb=GaussianNB()
mnb=MultinomialNB()

#TRAINING AND TESTING
x=df.drop('MEDV',axis=1)
y=df['MEDV']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
lreg.fit(x_train,y_train)
y_pred=lreg.predict(x_test)
print("Linear REGRESSION",mean_squared_error(y_test, y_pred))

rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print("RANDOM FOREST",mean_squared_error(y_test, y_pred))
#
gb.fit(x_train,y_train)
y_pred=gb.predict(x_test)
print("GRADIENT BOOSTING",mean_squared_error(y_test, y_pred))
#
dc.fit(x_train,y_train)
y_pred=dc.predict(x_test)
print("DECISION TREE",mean_squared_error(y_test, y_pred))
#
sv.fit(x_train,y_train)
y_pred=sv.predict(x_test)
print("SVR",mean_squared_error(y_test, y_pred))
#
mlp.fit(x_train,y_train)
y_pred=mlp.predict(x_test)
print("MLP",mean_squared_error(y_test, y_pred))
#

'''
Linear REGRESSION 19.98962953759516
RANDOM FOREST 13.828940081632643
GRADIENT BOOSTING 23.22202398031392
DECISION TREE 21.511938775510206
SVR 57.220989743865104
MLP 78.54502316743023
'''