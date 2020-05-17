#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# In[3]:


df = pd.read_csv('train.csv')

#fiiliing NaN values 
# In[4]:


df.isna().sum()


# In[5]:


df.Type_of_Cab.value_counts().plot(kind='bar')


# In[6]:


df.Type_of_Cab.fillna(method='ffill',inplace =True)


# In[7]:


df.isna().sum()


# In[8]:


df.Customer_Since_Months.value_counts().plot(kind='bar')


# In[9]:


df.Customer_Since_Months.fillna(method='ffill').value_counts().plot(kind='bar')  
#this look good for fractions for filling the NAN values


# In[10]:


df.Customer_Since_Months.fillna(method='ffill',inplace=True)


# In[11]:


df.Life_Style_Index.plot.hist()


# In[12]:


df.Life_Style_Index.fillna(pd.Series(np.random.uniform(1.596380, 4.875110, size=len(df)), index=df.index)).plot.hist()
#Filling randomly in range of Life Style Index looks like reasonable


# In[13]:


df.Life_Style_Index.fillna(pd.Series(np.random.uniform(1.596380, 4.875110, size=len(df)), index=df.index),inplace=True)


# In[14]:


df.Confidence_Life_Style_Index.value_counts().plot(kind='bar')


# In[15]:


df.Confidence_Life_Style_Index.fillna(method='ffill').value_counts().plot(kind='bar')
#Looks reasonable filling with method


# In[16]:


df.Confidence_Life_Style_Index.fillna(method='ffill',inplace=True)


# In[17]:


df.Var1.plot.hist()


# In[18]:


df.Var1.describe()


# In[19]:


df.Var1.fillna(method='ffill').plot.hist()
#better choice for filling missing data
df.Var1.fillna(method='ffill',inplace=True)


# In[21]:


df.head()


# In[35]:


Type_cab = pd.crosstab(df['Gender'],df['Surge_Pricing_Type'])
Type_cab.div(Type_cab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
#Customer Since month, gender


# In[37]:


def func(df):
    Type_of_cab = pd.get_dummies(df.Type_of_Cab).drop('E',axis=1)
    Confidence_life_style = pd.get_dummies(df.Confidence_Life_Style_Index).drop('C',axis=1)
    Destionation_type = pd.get_dummies(df.Destination_Type).drop('N',axis=1)
    Gender = pd.get_dummies(df.Gender).drop('Male',axis=1)
    new_df = df.drop(['Type_of_Cab','Confidence_Life_Style_Index','Destination_Type','Gender','Trip_ID','Var1','Var2','Var3'],axis=1)
    data = pd.concat([new_df,Type_of_cab,Confidence_life_style,Destionation_type,Gender],axis=1)
    return data 


# In[38]:


X = func(df).drop(['Surge_Pricing_Type','Customer_Since_Months','Female'],axis=1)
y = df.Surge_Pricing_Type


# In[39]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[85]:


x_train, x_test,y_train,y_test = train_test_split(X,y ,test_size = 0.2 ,random_state = 1)


# In[86]:


clf = DecisionTreeClassifier() 
model = clf.fit(x_train,y_train)
model


# In[87]:


y_pred = model.predict(x_test)
accuracy_score(y_test,y_pred)


# In[44]:


test_df = pd.read_csv('test.csv')


# In[46]:


test_df.isna().sum()


# In[54]:


test_df.Type_of_Cab.fillna(method='ffill',inplace =True)
test_df.Customer_Since_Months.fillna(method='ffill',inplace=True)
test_df.Life_Style_Index.fillna(pd.Series(np.random.uniform(1.596380, 4.875110, size=len(df)), index=df.index),inplace=True)
test_df.Confidence_Life_Style_Index.fillna(method='ffill',inplace=True)


# In[55]:


final = func(test_df).drop(['Customer_Since_Months','Female'],axis=1)


# In[57]:


test_df['Surge_Pricing_Type']=model.predict(final)


# In[59]:


result = test_df[['Trip_ID','Surge_Pricing_Type']]
result.to_csv('finalsubmission.csv',index =False)

