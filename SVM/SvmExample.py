# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 01:52:23 2019

@author: dat
"""

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X = np.array([[-1, -1], [-2, -1], [-2, 1], [2, 1]])
y = np.array([1, 1, -1, -1])

clf = SVC(C=1.0, kernel='linear')
clf.fit(X, y) 

new_point = np.array ( [[-0.5, -0.5], [0.5,0.5]] ) 
print ('\nuse predict')
print(clf.predict(new_point))

print ('\nuse decision_function')
print(clf.decision_function(new_point))


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_title(title)
ax.plot(new_point[:,0], new_point[:,1],'og')
plt.show()
