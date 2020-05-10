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

def sfs(answerVar, predictorVar, D):
    
    CV = 3
    hitRate = 'balanced_accuracy'
    
    solucionActual = pd.DataFrame()
    
    supportDataFrame = predictorVar
    
    if D is None:
        D = len(predictorVar.columns)
     
    k=1
    while(k<1+1):
        
        valPerIteration = dict()
        
        for j in range(0, len(supportDataFrame.columns)):
            
            if(k==1):
                X = supportDataFrame.iloc[:,j]
            else:
                X = pd.concat([supportDataFrame.iloc[:,j], solucionActual], axis=1, sort=False)

            
            y = answerVar
            validation = validacionRobusta(X.to_numpy().reshape(-1,1),y,CV,hitRate)
            valPerIteration.update({j:validation})
        
        validaciones = valPerIteration.values()
        bestCrossValue = max(validaciones)
        
        #Sacamos la key de los atributos que poseen el best cross value calculado anteriormente
        validKey = sacaKey(valPerIteration, bestCrossValue)
        #Obtenemos la columna de la mejor variable y la añadimos a la solución actual
        columnNameValid = supportDataFrame.columns[validKey]
        #Actualizamos Solucion Temporal
        solucionActual = pd.concat([solucionActual, supportDataFrame.iloc[:,validKey]], axis=1)
        #Quitamos la columna del conjunto de columnas a evaluar
        supportDataFrame = supportDataFrame.drop(supportDataFrame.columns[[validKey]], axis='columns')
        
        k = k+1
        
    return solucionActual
    

def sacaKey(varPerJ, bestCrossValue):

    for i in varPerJ:
        if varPerJ.get(i) == bestCrossValue:
            return i;
    

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