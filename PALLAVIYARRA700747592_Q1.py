#!/usr/bin/env python
# coding: utf-8

# In[26]:


#Question1
#imported necessary libraries and modules required
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# In[27]:


#read the csv data download and assigned it to a variable df
df=pd.read_csv('Downloads/NNDL_Code and Data/glass.csv')


# In[28]:


#checking what data is present
df


# In[29]:


#splitting data into features and target variable
X = df.drop('Type',axis=1) 
print(X)


# In[30]:


#splitting data into features and target variable
y = df['Type']
print(y)


# In[31]:


#splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# In[33]:


#training the model on training set
model = GaussianNB()  
model.fit(X_train, y_train)


# In[34]:


#predicting based on test set
y_pred = model.predict(X_test) 


# In[35]:


#measuring accuracy using test data and predicted data
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Score:", accuracy)


# In[36]:


#printing classification report
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))


# In[ ]:




