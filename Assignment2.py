import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#df = pd.read_csv("/Users/aaangd/Downloads/HMI-cancer-data.tsv" , sep = '\t')
df = pd.read_csv("HMI-cancer-data.tsv" , sep = '\t')
df2 = df.fillna(df.mode().iloc[0])
df2['Patient Status'].value_counts()
y = df2['Patient Status']
y.value_counts()
X= df2.drop('Patient Status' , axis =1)
X=(X-X.min())/(X.max()-X.min())
y.value_counts()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
y= label_encoder.fit_transform(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
logisticRegr = LogisticRegression(max_iter=1400)
logisticRegr.fit(x_train, y_train)
y_predict = logisticRegr.predict(x_test)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_predict)
#print(cm)
from sklearn.model_selection import cross_val_score
K_logic = cross_val_score(logisticRegr,X,y,cv=5)
print ("The accuracies of the logistic Regression models are", K_logic)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
K_knn = cross_val_score(knn,X,y,cv=5)
print("The accuracies of the KNN models are:", K_knn)
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(x_test)
K_gnb = cross_val_score(gnb,X,y,cv=5)
print ("The accuracies of GNB are:", K_gnb)
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

print ("The accuracies of SVM are:", clf)

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
K_svm = cross_val_score(clf,X,y,cv=5)
print (K_svm)
plt.title("Logistic Regression")
x_cv=['CV1','CV2','CV3','CV4','CV5']
plt.plot(x_cv,K_logic,color='red')
plt.show()
plt.title("KNN")
plt.plot(x_cv,K_knn,color='BLUE')
plt.show()
plt.title("GNB")
plt.plot(x_cv,K_gnb,color='GREEN')
plt.show()
plt.title("SVM")
plt.plot(x_cv,K_svm,color='orange')
plt.show()