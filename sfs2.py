import pandas as pd
import numpy as np
import math
import sklearn
import collections
import operator


import matplotlib.pyplot as plt
from sklearn import tree, linear_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

##Sacamos los datos del fichero csv
datos = pd.read_csv('titanic.csv', sep=',', index_col=0)
dataFrame = pd.DataFrame(datos)

#Devolver tabla con variables por k y su respectivo rendimiento
def sfs(answerVar, predictorVar, D):
    CV = 3
    hitRate = 'balanced_accuracy'
    
    solucionActual = pd.DataFrame()
    
    solucion = dict()
    
    tabla = pd.DataFrame()
    
    supportDataFrame = predictorVar
   
    if D is None:
        D = len(predictorVar.columns)
     
    k=1
    while(k<D+1):
        
        lastValue = 0
        solucionTemporal = pd.DataFrame();
        
        for j in range(0, len(supportDataFrame.columns)):
            X = pd.concat([solucionActual,supportDataFrame.iloc[:,j]], axis=1)
            y = answerVar
            if k==1:
                validation = validacionRobusta(X.to_numpy().reshape(-1,1),y,CV,hitRate)
            else:
                validation = validacionRobusta(X,y,CV,hitRate)
                
            if(validation>lastValue):
                lastValue = validation
                solucionTemporal = X
        
        #Eliminamos del supportDataFrame la mejor variable
        bestColumn = solucionTemporal.columns[k-1]
        del supportDataFrame[bestColumn]
        
        
        #Actualizamos la solución actual
        solucionActual = solucionTemporal
        solucion.update({tuple(solucionActual.columns):validation})
       
        k = k+1
        
        #Creamos un DataFrame para mostrar los resultados
        sorted(solucion.items(),key=operator.itemgetter(1), reverse=True)
       
        df = pd.DataFrame(solucion)
        print(df)
        

    return df

def sffs(answerVar, predictorVar):
    CV = 3
    hitRate = 'balanced_accuracy'
    solucionActual = pd.DataFrame()
    supportDataFrame = predictorVar
    añadidos = pd.DataFrame()
    eliminados = pd.DataFrame()
    
    
    k=1
    while(k<len(predictorVar.columns)+1):
        
        
        lastValue1 = 0
        solucionTemporal = pd.DataFrame();
        
        for j in range(0, len(supportDataFrame.columns)):
            X = pd.concat([solucionActual,supportDataFrame.iloc[:,j]], axis=1)
            y = answerVar
            if k==1:
                validation = validacionRobusta(X.to_numpy().reshape(-1,1),y,CV,hitRate)
            else:
                validation = validacionRobusta(X,y,CV,hitRate)
                
            if(validation>lastValue1):
                lastValue1 = validation
                solucionTemporal = X
        
        añadidos=solucionTemporal
        solucionActual=solucionTemporal
        if k!=1:
            lastValue2 = 0
            for j in range(0, len(solucionActual.columns)):
                solucionTemporal=solucionActual
                solucionTemporal=solucionTemporal.drop([solucionTemporal.columns[j]], axis=1)
                X = solucionTemporal
                y = answerVar
                if len(solucionActual.columns) == 2:
                    validation = validacionRobusta(X.to_numpy().reshape(-1,1),y,CV,hitRate)
                else:
                    validation = validacionRobusta(X,y,CV,hitRate)
                
                if(validation>lastValue2):
                    lastValue2 = validation
                    mejorSolucionTemporal = X
                    eliminadoTemporal = solucionActual.iloc[:,j]
            
            if lastValue2>lastValue1:
                solucionActual = mejorSolucionTemporal     
                eliminados=pd.concat([eliminados,eliminadoTemporal], axis=1)   
        
        #Eliminamos del supportDataFrame la mejor variable
        for c in range (0, len(añadidos.columns)):
            nombreColumna = añadidos.columns[c]
            if nombreColumna in supportDataFrame.columns:
                supportDataFrame=supportDataFrame.drop(nombreColumna, axis=1)
                

        print(solucionActual.columns)
        print(validation)
    
        k = k+1
        
    return solucionActual
    
    
    

    

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
#sffs(answerVar, predictorVar)
