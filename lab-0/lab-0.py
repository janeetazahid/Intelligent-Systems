####LAb 0

## IMPORTS ##
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import scale 
from sklearn.cross_validation import train_test_split
from sklearn import cluster
from sklearn import metrics
from sklearn import svm

##EXPLORING DATA##

#import data
digits = datasets.load_digits()
#digits data
digits_data=digits.data
print(digits_data.shape) #shape
#targets data
digits_target=digits.target
print(digits_target.shape) #shape
#number of unique labels 
numbers_digits=len(np.unique(digits.target)) 
print(numbers_digits)
#isolate images 
digits_images=digits.images
print(digits_images.shape)

##VISULIZE DATA 
fig=plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

#plt.show()

## PCA ##
randomized_pca=PCA(n_components=2)
reduced_data_rpca=randomized_pca.fit_transform(digits.data)
pca=PCA(n_components=2)
reduced_data_pca=pca.fit_transform(digits.data)
print(reduced_data_pca.shape)
print(reduced_data_rpca)
print(reduced_data_pca)

## PREPROCESSING ##
data=scale(digits.data)
X_train, X_test,y_train,y_test,images_train,images_test=train_test_split(data,digits.target,digits.images,test_size=0.25,random_state=42)

n_samples,n_features=X_train.shape
print(n_samples)
print(n_features)
n_digits=len(np.unique(y_train))
print(len(y_train))
print(len(X_test))

## K-means ##
clf=cluster.KMeans(init='k-means++',n_clusters=10,random_state=42)
clf.fit(X_train)
y_pred=clf.predict(X_test)
print(y_pred[:100])
print(y_test[:100])
print(clf.cluster_centers_.shape)
print(metrics.confusion_matrix(y_test,y_pred))

## SVM ##
X_train_v,X_test_v,y_train_v,y_test_v,images_train_v,images_test_v=train_test_split(digits.data,digits_target,digits.images,test_size=0.25,random_state=42)
#create model
svc_model=svm.SVC(gamma=0.001,C=100.,kernel='linear')
#fit data 
svc_model.fit(X_train_v,y_train_v)
svm.SVC(C=10,kernel='rbf',gamma=0.001).fit(X_train_v,y_train_v).score(X_test_v,y_test_v)
print(svc_model.predict(X_test_v))
print(y_test_v)
predicted=svc_model.predict(X_test_v)
print(metrics.classification_report(y_test_v,predicted))
print(metrics.accuracy_score(y_test_v,predicted))