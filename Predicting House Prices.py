#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
df = pd.read_csv("C:\\Users\\user\\Downloads\\house data\\data.csv")
df.head()
df.shape
df.info()
df.drop(['date', 'view', 'sqft_above', 'sqft_basement', 'street', 'statezip'], axis=1, inplace=True)


# In[23]:


bedroom=[1,2,3,4,5,6]
for i in bedroom:
    subset = df[df['bedrooms'] == i]
    sns.distplot(subset['price'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = i)
plt.legend(prop={'size': 16}, title = 'bedroom')
plt.title('Density Plot of bedrooms')
plt.xlabel('Delay (min)')
plt.ylabel('Density')


# In[24]:


sns.scatterplot(data=df, x='price', y='sqft_lot')


# In[25]:


sns.pairplot(df)
plt.show()


# In[26]:


df['city']= pd.factorize(df['city'])[0]
df['country']= pd.factorize(df['country'])[0]
correl = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correl, annot=True, cmap='rocket_r', linewidths=0.5, fmt='.2f')
b = (df.columns)
scaler = StandardScaler()
df = scaler.fit_transform(df)
df
df = pd.DataFrame(df,columns=b)
df.head()
x = df.drop('price',axis=1)
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[27]:


from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
mean_squared_error(y_pred,y_test)
Score = r2_score(y_pred,y_test)
Score
model.score(x_test,y_test)
print("MEAN SQUARE ERROR : ",mean_squared_error(y_test,y_pred))
print("ROOT MEAN SQUARE ERROR : ",np.sqrt(mean_squared_error(y_test,y_pred)))
print("MEAN ABSOLUTE ERROR : ",mean_absolute_error(y_test,y_pred))
print("TEST R2SCORE : ",r2_score(y_test,y_pred))


# In[ ]:




