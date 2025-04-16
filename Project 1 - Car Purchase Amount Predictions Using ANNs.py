#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENT

# You are working as a car salesman and you would like to develop a model to predict the total dollar amount that customers are willing to pay given the following attributes: 
# - Customer Name
# - Customer e-mail
# - Country
# - Gender
# - Age
# - Annual Salary 
# - Credit Card Debt 
# - Net Worth 
# 
# The model should predict: 
# - Car Purchase Amount 

# # STEP #0: LIBRARIES IMPORT
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # STEP #1: IMPORT DATASET

# In[2]:


car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')


# In[3]:


car_df


# # STEP #2: VISUALIZE DATASET

# In[4]:


sns.pairplot(car_df)


# # STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# In[6]:


X = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)


# In[ ]:





# In[7]:


X


# In[8]:


y = car_df['Car Purchase Amount']
y.shape


# In[9]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[10]:


scaler.data_max_


# In[11]:


scaler.data_min_


# In[12]:


print(X_scaled[:,0])


# In[13]:


y.shape


# In[14]:


y = y.values.reshape(-1,1)


# In[15]:


y.shape


# In[16]:


y_scaled = scaler.fit_transform(y)


# In[17]:


y_scaled


# # STEP#4: TRAINING THE MODEL

# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)


# In[19]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


# In[20]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[21]:


epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)


# # STEP#5: EVALUATING THE MODEL 

# In[24]:


print(epochs_hist.history.keys())


# In[25]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])


# In[26]:


# Gender, Age, Annual Salary, Credit Card Debt, Net Worth

X_Testing = np.array([[1, 50, 50000, 10985, 629312]])


# In[27]:


y_predict = model.predict(X_Testing)
y_predict.shape


# In[28]:


print('Expected Purchase Amount=', y_predict[:,0])


# In[ ]:




