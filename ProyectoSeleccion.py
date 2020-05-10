# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:08:27 2020

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



#SFS!!!!!
def sfs(answerVar, predictorVar, D):
    
    CV = 3
    hitRate = 'balanced_accuracy'
    varPerIter = dict()
    
    #Creamos un nuevo dataframe para guardas las soluciones de cada iteracion
    solucionActual = pd.DataFrame()
    var = pd.DataFrame()
    
    if D is None:
        D = len(predictorVar.columns)

    for k in range(0,D):
        
        varPerJ = dict()
        
        
        if k==0:
    
            for j in range(0, len(predictorVar.columns)):
                X = predictorVar.iloc[:,j]
                y = answerVar
                validacion = validacionRobusta(X.to_numpy().reshape(-1,1),y,CV,hitRate)
                varPerJ.update({j:validacion})
                
            
            validaciones = varPerJ.values()
            bestCrossValue = max(validaciones)
            validKey = sacaKey(varPerJ, bestCrossValue)
            columnNameValid = predictorVar.columns[validKey]
            solucionActual.insert(loc=0, column= columnNameValid,value = predictorVar.iloc[:,validKey])
            
            var = predictorVar.drop(predictorVar.columns[[validKey]], axis='columns')
        else:  
            for j in range(0,len(var.columns)):
                X = pd.concat([solucionActual,var.iloc[:,j]], axis=1)
                y = answerVar
                validacion = validacionRobusta(X,y,CV,hitRate)
                varPerJ.update({j:validacion})
                
            validaciones = varPerJ.values()
            bestCrossValue = max(validaciones)
            validKey = sacaKey(varPerJ, bestCrossValue)
            columnNameValid = var.columns[validKey]
            solucionActual.insert(loc=0, column= columnNameValid,value = var.iloc[:,validKey])
            
            var.drop(var.columns[[validKey]], axis='columns')
            
        
    return 0


def sacaKey(varPerJ, bestCrossValue):

    for i in varPerJ:
        if varPerJ.get(i) == bestCrossValue:
            return i;
    

#Validación robusta
#Seleccionamos el número de datos de entrada

def validacionRobusta(X,y,CV,hitRate):
    
    n_exp = 10  
  
    scoreI = list()
    accuracies = dict()

    for i in range(n_exp):
        tree = DecisionTreeClassifier()
        
        #Aplicamos validación cruzada
        scores = cross_val_score(tree,X,y,cv=CV,scoring=hitRate)
        #print(scores)
        avgScores = np.mean(scores)
        accuracies.update({i:avgScores})

    avgFinal = sum(accuracies.values())/n_exp
    return avgFinal



##TEST

answerVar = dataFrame.iloc[:,-1]
predictorVar = dataFrame.iloc[:,0:-1]
D=None

sfs(answerVar, predictorVar, D)
