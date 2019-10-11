
import sys,os,pickle
import numpy as np 
from copy import deepcopy

import math
import csv
from collections import Counter

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import datasets ## get some example small data from sklearn 


diabetes = datasets.load_diabetes() # a dictionary 
print(diabetes['DESCR']) ## show you detail of the data. 

labels = np.array ( diabetes['target'] ) 
## let's convert @labels into 0/1 
median_value = np.quantile(labels,0.5)
labels_01 = deepcopy(labels) ## must use @deepcopy otherwise you will copy by pointers and not values. ## see https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/
labels_01[labels<median_value] = 0 
labels_01[labels>=median_value] = 1

features = diabetes['data']
print (features[0:10]) ## see first 10 people 

## number of people 
num_people = features.shape[0]


def get_prob_01 (labels_01): ## !! based on @labels_01
  prob1 = sum(labels_01==1)/ len(labels_01) ## probability of seeing a 1 in @labels_01
  prob0 = sum(labels_01==0)/ len(labels_01) ## probability of seeing a 0 in @labels_01
  return [prob0,prob1]


## !! what is entropy of the whole data ? 
import scipy 
prob = get_prob_01 (labels_01)
data_entropy = scipy.stats.entropy ( prob )
print ('\nentropy of non-split data {}\n'.format(data_entropy))

## let's split 
def do_1_split (features,labels_01,which_col,data_entropy) :
  in_group_0 = features[:,which_col] < 0 ## labels saying if a person is in group less than 0
  in_group_1 = features[:,which_col] >= 0
  num_people_in0 = np.sum(in_group_0) ## count how many people in this group 0
  num_people_in1 = np.sum(in_group_1)
  entropy_group_0 = scipy.stats.entropy ( get_prob_01 ( labels_01[in_group_0] ) )
  entropy_group_1 = scipy.stats.entropy ( get_prob_01 ( labels_01[in_group_1] ) )
  expected_entropy_this_split = num_people_in0/num_people * entropy_group_0 + num_people_in1/num_people * entropy_group_1 
  information_gain = data_entropy - expected_entropy_this_split
  print ('information_gain is {}'.format(information_gain)) 


## let's experiment only the first 4 features Age, Sex, Body mass index, Average blood pressure. 
do_1_split (features,labels_01,0,data_entropy) ## let's try split the data by "Age" (column 0 in @features)
do_1_split (features,labels_01,3,data_entropy) ## let's try split the data by "Body mass index" (column 3 in @features)
print ('Body mass index is the better split than Age')


"""
Now we use the sklearn to build the tree on this data
The tree may not be perfect.
"""

my_tree = DecisionTreeClassifier (criterion='entropy',splitter='best',max_depth=4) ## use only 4 split 

## we manually split the data into 2 sets 
X_train, X_test, y_train, y_test = train_test_split(features, labels_01, test_size=0.3, random_state=1234)
my_tree.fit(X_train,y_train)
y_predict = my_tree.predict(X_test)
print ('accuracy {}'.format(sum ( y_predict==y_test ) / len(y_test)) )

sklearn.tree.plot_tree(my_tree) ## see the tree we just made 


## use 10-fold cross-validation, instead of one fold that we did manually
ten_fold_acc_tree = cross_val_score ( my_tree, X=features, y=labels_01, cv=10, scoring='accuracy' )


"""
Now we use KNeighborsClassifier
"""

my_knn = KNeighborsClassifier (n_neighbors = 10) ## take 10 closet neighbors

## use 10-fold cross-validation, instead of one fold that we did manually
ten_fold_acc_knn = cross_val_score ( my_knn, X=features, y=labels_01, cv=10, scoring='accuracy' )


"""
What method seems to be better here?
"""
print ('dec. tree gives mean accurracy and standard deviation {}, {}\n'.format( np.mean(ten_fold_acc_tree),np.std(ten_fold_acc_tree)))

print ('k-nearest nei. gives mean accurracy and standard deviation {}, {}\n'.format( np.mean(ten_fold_acc_knn),np.std(ten_fold_acc_knn)))

print ('range overlaps so, both methods are about the same.')

