
# coding: utf-8

# In[151]:


get_ipython().magic('matplotlib inline')
import os
import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("/home/perlzuser/Documents/krishna/AI/datasets/diabetes.csv")
#df.head()
#sns.pairplot(df,hue="Outcome")


# In[152]:


newdf=df.drop(['Age','BloodPressure','SkinThickness','BMI','Insulin'], axis = 1)
#newdf.head()
#ax = sns.barplot(x="Outcome", y="Pregnancies", data=df)
#sns.pairplot(newdf,hue="Outcome")


# In[153]:


newdf.loc[newdf['Pregnancies'] <= 3.7, 'Pregnancies'] = 0
newdf.loc[newdf['Pregnancies'] > 3.7, 'Pregnancies'] = 1


newdf.loc[newdf['Glucose'] < 100, 'Glucose'] = 0
newdf.loc[newdf['Glucose'] >= 100, 'Glucose'] = 1
#newdf.head()


# In[154]:


newdf.loc[newdf['DiabetesPedigreeFunction'] < .5, 'DiabetesPedigreeFunction'] = 0
newdf.loc[newdf['DiabetesPedigreeFunction'] >= .501, 'DiabetesPedigreeFunction'] = 1
#temp=newdf.copy()


# In[155]:


newdf['DiabetesPedigreeFunction'] = newdf['DiabetesPedigreeFunction'].astype(int)


# In[156]:


features=newdf[['Glucose','DiabetesPedigreeFunction']].copy()


# In[157]:


label=newdf[['Outcome']].copy()


# In[158]:


X_train,X_test,y_train,y_test = train_test_split(features,label, test_size = .5)


# In[159]:


clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[160]:


predictions = clf.predict(X_test)
print (accuracy_score(y_test, predictions))

