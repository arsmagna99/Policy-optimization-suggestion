#!/usr/bin/env python
# coding: utf-8

# In[1]: 属性


#Data attributes
#Proportion = "女性劳动者比例"
#Working years = "女性平均连续工作年数"
#Parental leave = "女性产假获取率"
#Overtime hours = "一个月平均加班时间" 
#Annual leave = "年假获取率"
#Management proportion = "管理职位女性比例"


# In[2]: 数据加工


# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from dtreeviz.trees import * #Import visualization module for Decision Tree

#Load and process data
df = pd.read_csv("woman.csv")
data = df.dropna(how='any')
df1 = data.round().astype(int)
df1.loc[df1['Overtime_hours(h)'] > 100, 'Overtime_hours(h)'] = 100
df1.loc[df1['Annual_leave(%)'] > 100, 'Annual_leave(%)'] = 100
df1.loc[df1['Parental_leave(%)'] > 100, 'Parental_leave(%)'] = 100
df1.loc[df1['Management_proportion(%)'] < 30, 'Management_proportion(%)'] = 0    
df1.loc[df1['Management_proportion(%)'] >= 30, 'Management_proportion(%)'] = 1

#Split dataset in independent and dependent variable
x = df1.loc[ : , df1.columns != 'Management_proportion(%)']
y = df1['Management_proportion(%)']


# In[3]: 建立模型


# Split dataset into training set and test set（70% training and 30% test）
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 

# Create Decision Tree Classifer object
clf = tree.DecisionTreeClassifier(criterion="gini",max_depth=4)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[4]: 可视化


#Visualize Decision Tree
clf.fit(x, y)
viz = dtreeviz(clf, 
               x, 
               y,
               target_name = 'Management_proportion(%)',
               feature_names = df1.columns, 
               class_names = ['low', 'high'],
               orientation ='LR',
               title = '决策树可视化',
               title_fontsize = 23
              )  
viz.save("result1.svg")   

#You can open result1.svg from your folder
