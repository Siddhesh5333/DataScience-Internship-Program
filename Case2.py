from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
df=pd.read_csv("C:/Users/CC-082/PycharmProjects/pythonProject/venv/DSINTERNSHIP/IRIS.csv")
x=df.drop('species',axis=1)
y=df['species']

#Feature Selection 1
'''
bestfeatures=SelectKBest(score_func=chi2,k='all')
fit=bestfeatures.fit(x,y)
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)
featuresScores=pd.concat([dfcolumns,dfscores],axis=1)
featuresScores.columns=['Specs','Score']
print(featuresScores)
'''

#Feature Selection 2
'''
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importance=pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()
'''

#Numerical to Categorical
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

rf=RandomForestClassifier()
df['sepal_length']=pd.cut(df['sepal_length'],3,labels=['0','1','2'])
df['sepal_width']=pd.cut(df['sepal_width'],3,labels=['0','1','2'])
df['petal_length']=pd.cut(df['petal_length'],3,labels=['0','1','2'])
df['petal_width']=pd.cut(df['petal_width'],3,labels=['0','1','2'])

le=LabelEncoder()
le.fit(y)
y=le.transform(y)
print(y)
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print('Random Forest: ',accuracy_score(y_test,y_pred))
'''


#Random over sampling
'''
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)
X,y=ros.fit_resample(X,y)
# print(ros.fit_resample(X,y))
'''

# Synthetic minority oversampling
'''
from imblearn.over_sampling import SMOTE
sms=SMOTE(random_state=0)
X,y=sms.fit_resample(X,y)
# print(sms.fit_resample(X,y))
'''

# under sampling
'''
from  imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=0)
X,y=rus.fit_resample(X,y)
# print(rus.fit_resample(X,y))
'''

# identifying Outlier using interquantile range
'''
print(df['sepal_length'])
Q1=df['sepal_length'].quantile(0.25)
Q2=df['sepal_length'].quantile(0.75)

IQR= Q2-Q1
print(IQR)

upper=Q2+1.5*IQR
lower=Q1-1.5*IQR
print(upper)
print(lower)

out1=df[df['sepal_length'] < lower].values
out2=df[df['sepal_length']>upper].values

df['sepal_length'].replace(out1,lower,inplace=True)
df['sepal_length'].replace(out2,upper,inplace=True)

print(df['sepal_length'])
'''

#  Principal component analysis
'''
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logr=LogisticRegression()
pca=PCA(n_components=2)
pca.fit(X)
X=pca.transform(X)

print(X)
'''

#linear regression
'''
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

bos=load_boston()
reg=LinearRegression()
'''