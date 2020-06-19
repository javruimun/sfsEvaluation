import pandas as pd
import numpy as np
import operator
from collections import OrderedDict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

##Sacamos los datos del fichero csv
datos = pd.read_csv('Titanic.csv', sep=',')
#datos = pd.read_csv('BreastCancerDataset.csv', sep=',')
dataFrame = pd.DataFrame(datos)

#Devolver tabla con variables por k y su respectivo rendimiento
def sfs(answerVar, predictorVar, D):
    CV = 3
    hitRate = 'balanced_accuracy'
    
    solucionActual = pd.DataFrame()
    
    solucion = dict()
    
    
    supportDataFrame = predictorVar
   
    if D is None:
        D = len(predictorVar.columns)
     
    k=1
    while(k<=D):
    
        
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
        print(solucionActual.columns)
        k = k+1
        
     #Creamos un DataFrame para mostrar los resultados
    solucion=OrderedDict(sorted(solucion.items(),key=operator.itemgetter(1), reverse=True))
        
    df = pd.DataFrame([[key, solucion[key], len(key)] for key in solucion.keys()], columns=['Solution', 'Score','Size'])
    print(df)        
    return df
        
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
