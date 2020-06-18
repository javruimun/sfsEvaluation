import pandas as pd
import numpy as np
import operator
from collections import OrderedDict


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


##Sacamos los datos del fichero csv
datos = pd.read_csv('titanic.csv', sep=',')
dataFrame = pd.DataFrame(datos)



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




#Test para cross_val_score
X=dataFrame.iloc[:,0:-1]
y=dataFrame.iloc[:,-1]
tree = DecisionTreeClassifier()
CV = 3
hitRate = 'balanced_accuracy'

#Test para cross_val_score
print(cross_val_score(tree,X,y,cv=CV,scoring=hitRate))

#Test para validacionRobusta
#print(validacionRobusta(X,y,CV,hitRate))