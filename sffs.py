import pandas as pd
import numpy as np
import operator
from collections import OrderedDict


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


##Sacamos los datos del fichero csv
datos = pd.read_csv('BreastCancerDataset.csv', sep=',')
dataFrame = pd.DataFrame(datos)

#Creación del algoritmo SFFS
def sffs(answerVar, predictorVar):
    
    CV = 4
    hitRate = 'balanced_accuracy'
    
    solucionActual = pd.DataFrame()
    supportDataFrame = predictorVar
    
    añadidos = list()
    eliminados = list()
    solucion = dict()
    solucionFinal=dict()
    
    
    k=0
    
    while(k<len(predictorVar.columns)):
        
        mejorSolucionTemporal = seleccionaMejorVariable(supportDataFrame, solucionActual,k,CV,hitRate)
          
        solucionActual = mejorSolucionTemporal[0]
        rendimiento = mejorSolucionTemporal[1]
        añadidos.append(mejorSolucionTemporal[2])
        
        if k!=0:
            res = eliminaSiHayMejora(solucionActual, rendimiento, CV, hitRate, eliminados)
            solucionActual= res[0]
            eliminados= res[1]
            validation= res[2]
            print(validation)
            #Guardamos todas las soluciones en un diccionario
            solucion.update({tuple(solucionActual.columns):validation})
            
          #Eliminamos del supportDataFrame la mejor variable
        for c in range (0, len(añadidos)):
            nombreColumna = añadidos[c]
            if nombreColumna in supportDataFrame.columns:
                supportDataFrame=supportDataFrame.drop(nombreColumna, axis=1)
                #Comparamos lista
      
        
        print(solucionActual.columns)
        
        
        
        
        k=k+1
    ultimaSolucion = solucionActual
    ultimaValidation = validation
    #Creamos un DataFrame para mostrar los resultados de cada iteración
    solucionSinParada=mostrarSolucion(solucion)
    print(solucionSinParada)
    
    añadidosList= sorted(añadidos)
    predictorVarList= sorted(predictorVar.columns)
    
    if añadidosList == predictorVarList:
        res=condicionDeParada(eliminados,solucionActual,validation,CV, hitRate)
        solucionActual=res[0]
        rendimiento=res[1]
    
    solucionFinal.update({tuple(solucionActual.columns):rendimiento})
    solucionConParada=mostrarSolucion(solucionFinal)
    
    hayMejora = ultimaSolucion.columns.tolist() == list(solucionConParada['Solution'].values.tolist()[0]) and ultimaValidation == solucionConParada['Score'].values[0]
    
    if hayMejora:
        print('No se ha conseguido mejorar el resultado con la condicion de parada')
    else:
         print('Si se ha conseguido mejorar el resultado con la condicion de parada')
        
    
    print(solucionConParada)
    print(añadidos)
    return solucionActual

def seleccionaMejorVariable(supportDataFrame, solucionActual, k,CV,hitRate):
    
    lastValue = 0
    bestSolucionTemporal = pd.DataFrame()
    bestColumn = ''
    
    for j in range(0, len(supportDataFrame.columns)):
            
            solucionTemporal = supportDataFrame.iloc[:,j]
            X = pd.concat([solucionActual,solucionTemporal], axis=1)
            y = answerVar
            
            if k==0:
                validation = validacionRobusta(X.to_numpy().reshape(-1,1),y,CV,hitRate)
            else:
                validation = validacionRobusta(X,y,CV,hitRate)
                
            if(validation>lastValue):
                lastValue = validation
                bestSolucionTemporal = X
                bestColumn = solucionTemporal.name
                
    
    return (bestSolucionTemporal, lastValue, bestColumn)
    
    
def eliminaSiHayMejora(solucionActual, rendimiento, CV, hitRate, eliminados):
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
            colunmDeleted = solucionActual.iloc[:,j]
            eliminadoTemporal = colunmDeleted.name
        
    if lastValue2>=rendimiento:
        validation=lastValue2
        solucionActual = mejorSolucionTemporal     
        eliminados.append(eliminadoTemporal)
    else:
        validation=rendimiento
    
    return (solucionActual, eliminados, validation)

def mostrarSolucion(solucion):
    solucionDataFrame = pd.DataFrame([[key, solucion[key], len(key)] for key in solucion.keys()], columns=['Solution', 'Score','Size'])
    solucionDataFrame = solucionDataFrame.sort_values('Score',ascending=False)
    
    return solucionDataFrame
 
def condicionDeParada(eliminados,solucionActual,validation,CV, hitRate):
    c=0
    eliminadosSinParada=eliminados[:]
    rendimiento = validation
    while(c<10):
        res = eliminaSiHayMejora(solucionActual, rendimiento, CV, hitRate, eliminados) 
        solucionActual = res[0]
        rendimiento = res[2]
        if eliminadosSinParada == eliminados:
            c=c+1
            
        else:
            c=0
            eliminadosSinParada = eliminados[:]
    return (solucionActual,rendimiento )

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

#TEST

answerVar = dataFrame.iloc[:,-1]
predictorVar = dataFrame.iloc[:,0:-1]
sffs(answerVar, predictorVar)
