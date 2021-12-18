# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:24:17 2021

@author: Sinem
"""

import pandas as pd
import numpy as np
import seaborn as sns
#diabets data
diabetes = pd.read_csv("diabetes.csv")
df=diabetes.copy()
print(df)


df.isna().sum()
#kolonların data type'ı verir
df.info()
df['Outcome'] = df['Outcome'].astype('category')
df.info()
df['Outcome'].value_counts()



#display values pychart
df['Outcome'].value_counts().plot.pie();

#(horizontal)
df['Outcome'].value_counts().plot.barh();

#(vertical)
df['Outcome'].value_counts().plot.bar();

#statistical description for numeric values
df.describe().T

#With boxplot, those with a z-score greater than 3 are outliers, we see them
import seaborn as sns
sns.boxplot(x='Pregnancies', data=df);

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)                                  
                                               
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 200)
model.fit(X_train, y_train)
#bias
model.intercept_
#weight
model.coef_
#prediction 
y_pred = model.predict(X_test)
y_pred

#x test
X_test[:5]

#y pred
y_pred[:5]

#model performance
from sklearn.metrics import confusion_matrix
#confusion matrix actual positive actual negative 
df = pd.DataFrame(confusion_matrix(y_test, y_pred), columns = ['Predicted Positive', 'Predicted Negative'], 
                  index=['Actual Positive', 'Actual Negative'])
print(df)

#accuracy 
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#classifaction yerine model performansı
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test, y_pred, ))
print("Recall:", recall_score(y_test,y_pred))
print("F1 Score:", f1_score(y_test,y_pred))



#xtest
X_test[:10]
#shows the probability, we get the ratio whichever is higher in the 1 row, we say diabetes or not.
model.predict_proba(X_test)[0:10]

# Actual Data
y_test[0:10]

#All probability estimates between 0 and 10
y_probability = model.predict_proba(X_test)
y_probability = y_probability[:,1]
y_probability[0:10]

# threshold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
model_roc_auc = roc_auc_score(y_test, model.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % model_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim(([0.0, 1.0]))
plt.ylim(([0.0, 1.05]))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()

#The closer the roc auc performance value is to being square, the better it performs.
print(model_roc_auc)





