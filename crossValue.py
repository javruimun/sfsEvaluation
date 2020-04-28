# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:32:08 2020

@author: Javier
"""

import pandas as pd
import numpy as np
import math
import sklearn

import matplotlib.pyplot as plt
from sklearn import tree, linear_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier



##Sacamos los datos del fichero csv
datos = pd.read_csv('titanic.csv', sep=',', index_col=0)
dataFrame = pd.DataFrame(datos)


#Validación cruzada
#Seleccionamos el número de datos de entrada
X = dataFrame.loc[:, ['Sex', 'Age', 'Fare']]
y = dataFrame.loc[:,['Survived']]


def validacionCruzada():
    
    scoreI = list()
    accuracies = dict()
    n_exp = 10
    
    for i in range(n_exp):
        tree = DecisionTreeClassifier()
        modelo = tree.fit(X,y)
        scores = cross_val_score(modelo,X,y,cv=3,
                             scoring = 'balanced_accuracy')
        avgScores = sum(scores)/len(scores)
        accuracies.update({i:avgScores})
        i+=1

    avgFinal = sum(accuracies.values())/n_exp
    return avgFinal


print(validacionCruzada())










