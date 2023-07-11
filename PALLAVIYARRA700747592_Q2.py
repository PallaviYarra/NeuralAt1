#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Question2
#imported necessary libraries and modules required
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning


# In[4]:


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# In[5]:


#read the csv data download and assigned it to a variable df
df=pd.read_csv('Downloads/NNDL_Code and Data/glass.csv')


# In[6]:


#checking what data is present
df


# In[7]:


#splitting data into features and target variable
X = df.drop('Type',axis=1) 
print(X)


# In[8]:


#splitting data into features and target variable
y = df['Type']
print(y)


# In[9]:


#splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# In[10]:


svm = LinearSVC(random_state=38)
svm.fit(X_train, y_train)


# In[11]:


score = svm.score(X_test, y_test)
y_pred = svm.predict(X_test)
report = classification_report(y_test, y_pred)


# In[12]:


print(f"Accuracy score: ", score)
print(f"Classification report:\n", report)


# In[ ]:





# In[ ]:





# In[ ]:




