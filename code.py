#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn matplotlib reportlab')


# In[2]:


import pandas as pd


# In[3]:


from sklearn.datasets import load_iris


# In[4]:


from sklearn.model_selection import train_test_split


# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[7]:


iris = load_iris()


# In[8]:


data.data


# In[9]:


iris.data


# In[10]:


X = iris.data


# In[11]:


y = iris.target


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


dt_classifier = DecisionTreeClassifier()


# In[14]:


dt_classifier.fit(X_train, y_train)


# In[15]:


y_pred_dt = dt_classifier.predict(X_test)


# In[16]:


accuracy_dt = accuracy_score(y_test, y_pred_dt)


# In[17]:





# In[21]:


accuracy_dt = accuracy_score(y_test, y_pred_dt)


# In[22]:


print("Decision Trees Accuracy:", accuracy_dt)


# In[23]:


from sklearn.svm import SVC


# In[24]:


svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)


# In[25]:


y_pred_svm = svm_classifier.predict(X_test)


# In[26]:


accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)


# In[27]:


from sklearn.neighbors import KNeighborsClassifier


# In[28]:


knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)


# In[29]:


y_pred_knn = knn_classifier.predict(X_test)


# In[30]:


accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)


# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)


# In[33]:


y_pred_rf = rf_classifier.predict(X_test)


# In[34]:


accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)


# In[35]:


from sklearn.cluster import KMeans


# In[36]:


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)


# In[37]:


import os
os.environ['OMP_NUM_THREADS'] = '1'

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)


# In[38]:


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)


# In[39]:


cluster_labels_kmeans = kmeans.labels_


# In[40]:


from sklearn.cluster import AgglomerativeClustering


# In[41]:


hierarchical = AgglomerativeClustering(n_clusters=3)
cluster_labels_hierarchical = hierarchical.fit_predict(X)


# In[42]:


from sklearn.mixture import GaussianMixture


# In[43]:


gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)


# In[44]:


cluster_labels_gmm = gmm.predict(X)


# In[45]:


import numpy as np


# In[46]:


num_states = 4  
num_actions = 2  
Q_table = np.zeros((num_states, num_actions))


# In[47]:


alpha = 0.1  
gamma = 0.9  
num_episodes = 1000


# In[48]:


for episode in range(num_episodes):
    state = 0  
    
    while state != 3:  
       
        if np.random.uniform(0, 1) < 0.1: 
            action = np.random.randint(0, num_actions)
        else:  
            action = np.argmax(Q_table[state])

        if action == 0: 
            next_state = max(state - 1, 0)
            reward = 0
        else:  
            next_state = min(state + 1, num_states - 1)
            reward = 1 if next_state == num_states - 1 else 0  
        
        Q_table[state, action] += alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action])
        
        state = next_state


# In[49]:


print("Learned Q-table:")
print(Q_table)


# In[50]:


import tensorflow as tf


# In[51]:


class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


# In[52]:


dqn_model = DQN(num_actions=num_actions)


# In[53]:


loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# In[54]:


dqn_model.compile(optimizer=optimizer, loss=loss_function)


# In[ ]:




